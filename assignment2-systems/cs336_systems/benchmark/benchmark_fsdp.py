import json
import statistics

import argparse
import torch
import timeit
import os

import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_systems.ddp import DDPModule
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


def run_step(device, inputs, targets, step, model, optimizer, compute_dtype, max_norm, lr_max, lr_min, t_w, t_c):
    optimizer.zero_grad(set_to_none=True)
    # FSDP already casts Linear/Embedding weights to compute_dtype; autocast handles the
    # remaining mixed-dtype ops (e.g. fp32 softmax output @ bf16 V).
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

    if isinstance(model, FSDPModule):
        clip_gradient_fsdp(model, max_norm)
    else:
        clip_gradient(model.parameters(), max_norm)
    for g in optimizer.param_groups:
        g["lr"] = get_cosine_lr(lr_max, lr_min, t_w, t_c, step + 1)

    optimizer.step()
    return t1 - t0  # exposed gradient sync time


def setup(rank, world_size, backend, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
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

    setup(rank, world_size, backend, args.master_port)

    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    is_cuda = device.startswith("cuda")
    print(f"[{rank}] {device=}")

    vocab_size = 10000
    context_length = args.context_length
    batch_size = args.batch_size
    rope_theta = 10000.0
    compute_dtype = DTYPES[args.compute_dtype]

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

    if args.fsdp:
        model = FSDPModule(model, compute_dtype=compute_dtype)
    else:
        model = DDPModule(model)
    optimizer = AdamW(model.parameters())

    inputs, targets = get_random_batch(batch_size, vocab_size, context_length, device)

    max_norm = 1.0
    lr_max = 1e-3
    lr_min = 1e-4
    t_w = 100
    t_c = 10

    mode = "fsdp" if args.fsdp else "ddp"

    # Exclude init (full unsharded model before FSDP sharding) from the peak: drop cached
    # blocks, then reset the peak counters so only training steps are measured.
    sync(device)
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        setup_allocated = torch.cuda.memory_allocated()
        if args.mem_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

    # First step records the fwd order (no prefetch); later steps prefetch.
    # The first step also allocates optimizer state, so it is included in the peak.
    print(f"[{rank}] Warmup...")
    for step in range(args.warmup_steps):
        run_step(device, inputs, targets, step, model, optimizer, compute_dtype, max_norm, lr_max, lr_min, t_w, t_c)

    sync(device)

    if is_cuda and args.mem_snapshot:
        if rank == 0:
            os.makedirs(args.mem_prof_dir, exist_ok=True)
            path = os.path.join(args.mem_prof_dir, f"{args.mem_prof_file}_{mode}_{args.compute_dtype}_ws{world_size}.pickle")
            # Load in https://pytorch.org/memory_viz
            torch.cuda.memory._dump_snapshot(path)
            print(f"[{rank}] Saved memory snapshot to {path}")
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"[{rank}] Benchmarking...")

    comm_times = []
    step_times = []

    for step in range(args.benchmark_steps):
        dist.barrier()
        t0 = timeit.default_timer()
        comm_time = run_step(device, inputs, targets, step, model, optimizer, compute_dtype, max_norm, lr_max, lr_min, t_w, t_c)
        sync(device)
        comm_times.append(comm_time)
        step_times.append(timeit.default_timer() - t0)

    if is_cuda:
        # [setup, after-step (params + grads + optimizer state), peak allocated, peak reserved], max over ranks
        mem = torch.tensor(
            [
                setup_allocated,
                torch.cuda.memory_allocated(),
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
            ],
            dtype=torch.float64,
            device=device,
        )
        print(f"[{rank}] Peak allocated: {gib(int(mem[2])):.2f} GiB, Peak reserved: {gib(int(mem[3])):.2f} GiB")
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        setup_gib, after_step_gib, peak_alloc_gib, peak_reserved_gib = (gib(int(v)) for v in mem.tolist())

    dist.barrier()

    if rank == 0:
        step_time = statistics.mean(step_times)
        comm_time = statistics.mean(comm_times)
        print(f"Done. Step Time:{step_time:.4f}, Comm Time:{comm_time:.4f}")
        if is_cuda:
            print(
                f"[{mode} ws={world_size}] setup: {setup_gib:.2f} GiB, after step: {after_step_gib:.2f} GiB, "
                f"peak allocated: {peak_alloc_gib:.2f} GiB, peak reserved: {peak_reserved_gib:.2f} GiB"
            )
            if args.results_file:
                with open(args.results_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "mode": mode,
                                "world_size": world_size,
                                "compute_dtype": args.compute_dtype,
                                "batch_size": batch_size,
                                "setup_gib": setup_gib,
                                "after_step_gib": after_step_gib,
                                "peak_allocated_gib": peak_alloc_gib,
                                "peak_reserved_gib": peak_reserved_gib,
                                "step_time_s": step_time,
                                "comm_time_s": comm_time,
                            }
                        )
                        + "\n"
                    )

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
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=True, help="--no-fsdp uses DDPModule")

    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--master_port", type=int, default=29500)

    parser.add_argument("--mem_prof_dir", type=str, default="/home/wilsonhong/gdrive/cs336")
    parser.add_argument("--mem_prof_file", type=str, default="memprof")
    parser.add_argument("--mem_snapshot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--results_file", type=str, default=None, help="Append a JSON line of results (rank 0)")
    args = parser.parse_args()

    mp.spawn(
        fn=benchmark,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
