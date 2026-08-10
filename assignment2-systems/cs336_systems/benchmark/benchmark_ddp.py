import statistics
from cs336_systems.ddp import DDPModule, NaiveDDPModule
import argparse
import torch
import timeit
import os

import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW, get_cosine_lr


# TODO: benchmark memory usage!


def get_random_batch(batch_size: int, vocab_size, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_tokens = [torch.randint(vocab_size, (context_length + 1,), dtype=torch.long, device=device) for _ in range(batch_size)]

    x = torch.stack([tokens[:context_length] for tokens in batch_tokens])
    y = torch.stack([tokens[1:] for tokens in batch_tokens])
    return x, y


def sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_step(device, inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c):
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)

    loss.backward()
    sync(device)
    t0 = timeit.default_timer()
    # model.reduce_all_gradients()  # Naive
    model.finish_gradient_synchronization()

    sync(device)
    t1 = timeit.default_timer()

    clip_gradient(model.parameters(), max_norm)
    for g in optimizer.param_groups:
        g["lr"] = get_cosine_lr(lr_max, lr_min, t_w, t_c, step + 1)

    optimizer.step()
    return t1 - t0  # gradient sync time


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_DEBUG"] = "WARN"
    if backend == "nccl":
        dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group(backend, rank=rank, world_size=world_size)


def shutdown():
    dist.destroy_process_group()


def benchmark(
    rank: int,
    args: argparse.Namespace,
):
    world_size = args.world_size
    backend = args.backend

    setup(rank, world_size, backend)

    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    print(f"[{rank}] {device=}")
    # print(device)

    vocab_size = 10000
    context_length = args.context_length
    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    rope_theta = 10000.0
    gradient_checkpointing = args.gradient_checkpointing
    layer_chunk_size = args.layer_chunk_size

    print(f"[{rank}] Init model...")
    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        gradient_checkpointing,
        layer_chunk_size,
    ).to(device)

    # model = NaiveDDPModule(model)
    model = DDPModule(model)

    optimizer = AdamW(model.parameters())  # use default values

    # reuse random inputs
    batch_size = args.batch_size

    inputs, targets = get_random_batch(batch_size, vocab_size, context_length, device)

    max_norm = 1.0
    lr_max = 1e-3
    lr_min = 1e-4
    t_w = 100
    t_c = 10

    print(f"[{rank}] Warmup...")
    for step in range(args.warmup_steps):
        run_step(device, inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c)

    sync(device)

    print(f"[{rank}] Benchmarking...")

    comm_times = []
    step_times = []

    for step in range(args.benchmark_steps):
        if rank == 0:
            print(f"[{rank}] {step=}")
        dist.barrier()
        t0 = timeit.default_timer()
        comm_time = run_step(device, inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c)
        sync(device)
        comm_times.append(comm_time)
        step_times.append(timeit.default_timer() - t0)

    if rank == 0:
        print(f"Done. Step Time:{statistics.mean(step_times):.4f}, Comm Time:{statistics.mean(comm_times):.4f}")

    try:
        shutdown()
    finally:
        pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM")

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)

    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=2560)
    parser.add_argument("--d_ff", type=int, default=10240)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer_chunk_size", type=int, default=1)

    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])

    args = parser.parse_args()

    mp.spawn(
        fn=benchmark,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
