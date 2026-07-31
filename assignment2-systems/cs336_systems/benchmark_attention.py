import statistics
import argparse

import torch

# import torch.cuda.nvtx as nvtx
import timeit

from cs336_basics.model import CausalMultiHeadSelfAttention, RotaryEmbedding


"""
uv run python -m cs336_systems.benchmark_attention
"""


def get_random_batch(batch_size: int, vocab_size, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_tokens = [torch.randint(vocab_size, (context_length + 1,), dtype=torch.long, device=device) for _ in range(batch_size)]

    x = torch.stack([tokens[:context_length] for tokens in batch_tokens])
    y = torch.stack([tokens[1:] for tokens in batch_tokens])
    return x, y


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_dtype(precision: str):
    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    return torch.float32


def benchmark(
    *,
    mem_prof_file: str,
    mixed_precision: str,
    batch_size: int = 8,
    warmup_steps: int,
    benchmark_steps: int,
    context_length: int,
    d_model: int,
    num_heads: int,
    torch_compile: bool,
    rope_theta: float | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)

    dtype = get_dtype(mixed_precision)

    d_head = d_model // num_heads
    positional_encoder = RotaryEmbedding(context_length, d_head, rope_theta) if rope_theta is not None else None

    with torch.autocast(device, dtype=dtype):
        # vocab_size = get_vocab_size(vocab_path)  # NOTE: keep it now
        # print("vocab_size:", vocab_size)
        # print("Init attention...")
        model = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            positional_encoder,
        ).to(device)

        if torch_compile:
            print("  torch.compile")
            model = torch.compile(model)

    # : generate inputs
    x = torch.rand((batch_size, context_length, d_model), dtype=torch.float32, device=device)

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # print("Warmup...")
    for step in range(warmup_steps):
        y = model(x)
        loss = y.sum()
        loss.backward()

    sync()

    forward_times = []
    backward_times = []

    # with torch.cuda.profiler.profile():
    # print("Benchmarking...")

    # Start recording memory history.

    for step in range(benchmark_steps):
        sync()
        t0 = timeit.default_timer()
        y = model.forward(x)
        sync()
        t1 = timeit.default_timer()

        loss = y.sum()

        sync()
        t2 = timeit.default_timer()
        loss.backward()
        sync()
        t3 = timeit.default_timer()

        forward_times.append(t1 - t0)
        backward_times.append(t3 - t2)

    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot(f"/home/wilsonhong/gdrive/cs336/memprof_attn_{mem_prof_file}.pickle")
    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"[Forward] mean={statistics.mean(forward_times):.6f}, std={statistics.stdev(forward_times):.6f}")
    print(f"[Backard] mean={statistics.mean(backward_times):.6f}, std={statistics.stdev(backward_times):.6f}")

    # print("Done")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM")

    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--benchmark_steps", type=int, default=100)

    # parser.add_argument("--d_model", type=int, default=16)
    # parser.add_argument("--context_length", type=int, default=256)
    # parser.add_argument("--mem_prof_file", type=str, default="test")
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("-mp", "--mixed_precision", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=False)

    # parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # benchmark(**vars(args))

    for d_model in [16, 32, 64, 128]:
        for context_length in [256, 1024, 4096, 8192, 16384]:
            print(f"#### {d_model=}, {context_length=}")
            args.d_model = d_model
            args.context_length = context_length
            args.mem_prof_file = f"compiled_dm_{d_model}_ctx_{context_length}"
            benchmark(**vars(args))


if __name__ == "__main__":
    main()
