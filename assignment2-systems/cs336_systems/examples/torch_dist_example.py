import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_DEBUG"] = "WARN"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def demo(rank, world_size, backend):
    setup(rank, world_size, backend)

    data = torch.randint(0, 10, (3,))
    if backend == "nccl":
        data = data.to(f"cuda:{rank}")

    print(f"rank {rank} data (before all-reduce: {data})")

    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")
    dist.destroy_process_group()


def benchmark(rank, world_size, data_size, backend):
    setup(rank, world_size, backend)

    data = torch.rand((data_size,), dtype=torch.float32)
    if backend == "nccl":
        data = data.to(f"cuda:{rank}")
    torch.cuda.synchronize()
    # warmup
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        dist.barrier()

    t0 = timeit.default_timer()
    for _ in range(10):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        dist.barrier()
    t1 = timeit.default_timer()
    if rank == 0:
        print(f"time: {t1 - t0}")

    dist.destroy_process_group()


if __name__ == "__main__":
    for world_size in [2, 4, 6]:
        for data_size in [1_000_000 // 4, 10_000_000 // 4, 100_000_000 // 4, 1_000_000_000 // 4]:
            print(f"{world_size=}  {data_size=} ")
            mp.spawn(
                fn=benchmark,
                args=(
                    world_size,
                    data_size,
                    "gloo",
                ),
                nprocs=world_size,
                join=True,
            )
