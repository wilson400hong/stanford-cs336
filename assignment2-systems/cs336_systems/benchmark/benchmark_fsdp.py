import statistics

import argparse
import torch
import timeit
import os

import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_systems.fsdp import FSDPModule


DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def get_random_batch(batch_size: int, vocab_size, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_tokens = [torch.randint(vocab_size, (context_length + 1,), dtype=torch.long, device=device) for _ in range(batch_size)]

    x = torch.stack([tokens[:context_length] for tokens in batch_tokens])
    y = torch.stack([tokens[1:] for tokens in batch_tokens])
    return x, y


def sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def clip_gradient_fsdp(model: FSDPModule, max_norm: float):
    """Global-norm clipping: sharded grads are summed across ranks, replicated grads are counted once."""
    device = next(model.parameters()).device
    sharded_sq = torch.zeros((), device=device)
    replicated_sq = torch.zeros((), device=device)
    for p, ps in model.param_states.items():
        if p.grad is None:
            continue
        sq = p.grad.float().pow(2).sum()
        if ps.is_shardable:
            sharded_sq += sq
        else:
            replicated_sq += sq

    dist.all_reduce(sharded_sq, op=dist.ReduceOp.SUM)
    norm = torch.sqrt(sharded_sq + replicated_sq)
    clip_coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for p in model.param_states:
        if p.grad is not None:
            p.grad.mul_(clip_coef)


def run_step(device, inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c):
    optimizer.zero_grad(set_to_none=True)
    # FSDP already casts Linear/Embedding weights to compute_dtype; autocast handles the
    # remaining mixed-dtype ops (e.g. fp32 softmax output @ bf16 V).
    compute_dtype = model.compute_dtype
    with torch.autocast(device_type=device.split(":")[0], dtype=compute_dtype, enabled=compute_dtype is not None):
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

    loss.backward()
    sync(device)
    t0 = timeit.default_timer()
    # Waits on the outstanding reduce-scatter / all-reduce launched from the grad hooks
    model.finish_gradient_synchronization()

    sync(device)
    t1 = timeit.default_timer()

    clip_gradient_fsdp(model, max_norm)
    for g in optimizer.param_groups:
        g["lr"] = get_cosine_lr(lr_max, lr_min, t_w, t_c, step + 1)

    optimizer.step()
    return t1 - t0  # exposed gradient sync time


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_DEBUG"] = "WARN"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))
    else:
        dist.init_process_group(backend, rank=rank, world_size=world_size)


def shutdown():
    dist.destroy_process_group()


def gib(n_bytes: int) -> float:
    return n_bytes / 1024**3


def benchmark(
    rank: int,
    args: argparse.Namespace,
):
    world_size = args.world_size
    backend = args.backend

    setup(rank, world_size, backend)

    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    is_cuda = device.startswith("cuda")
    print(f"[{rank}] {device=}")

    vocab_size = 10000
    context_length = args.context_length
    batch_size = args.batch_size
    rope_theta = 10000.0
    compute_dtype = DTYPES[args.compute_dtype]

    if is_cuda:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    print(f"[{rank}] Init model...")
    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        rope_theta,
    ).to(device)

    model = FSDPModule(model, compute_dtype=compute_dtype)
    optimizer = AdamW(model.parameters())

    inputs, targets = get_random_batch(batch_size, vocab_size, context_length, device)

    max_norm = 1.0
    lr_max = 1e-3
    lr_min = 1e-4
    t_w = 100
    t_c = 10

    # First step records the fwd order (no prefetch); later steps prefetch.
    print(f"[{rank}] Warmup...")
    for step in range(args.warmup_steps):
        run_step(device, inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c)

    sync(device)

    if is_cuda:
        if rank == 0:
            os.makedirs(args.mem_prof_dir, exist_ok=True)
            path = os.path.join(args.mem_prof_dir, f"{args.mem_prof_file}_{args.compute_dtype}_ws{world_size}.pickle")
            # Load in https://pytorch.org/memory_viz
            torch.cuda.memory._dump_snapshot(path)
            print(f"[{rank}] Saved memory snapshot to {path}")
        torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.reset_peak_memory_stats()

    print(f"[{rank}] Benchmarking...")

    comm_times = []
    step_times = []

    for step in range(args.benchmark_steps):
        dist.barrier()
        t0 = timeit.default_timer()
        comm_time = run_step(device, inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c)
        sync(device)
        comm_times.append(comm_time)
        step_times.append(timeit.default_timer() - t0)

    if is_cuda:
        print(
            f"[{rank}] Peak allocated: {gib(torch.cuda.max_memory_allocated()):.2f} GiB, "
            f"Peak reserved: {gib(torch.cuda.max_memory_reserved()):.2f} GiB"
        )
    dist.barrier()

    if rank == 0:
        print(f"Done. Step Time:{statistics.mean(step_times):.4f}, Comm Time:{statistics.mean(comm_times):.4f}")

    shutdown()


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM with FSDP")

    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--benchmark_steps", type=int, default=5)

    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=2560)
    parser.add_argument("--d_ff", type=int, default=10240)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--compute_dtype", type=str, default="fp32", choices=list(DTYPES))

    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])

    parser.add_argument("--mem_prof_dir", type=str, default="/home/wilsonhong/gdrive/cs336")
    parser.add_argument("--mem_prof_file", type=str, default="memprof_fsdp")
    args = parser.parse_args()

    mp.spawn(
        fn=benchmark,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
