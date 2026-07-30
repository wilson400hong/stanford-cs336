import statistics
import argparse

import torch

# import torch.cuda.nvtx as nvtx
import timeit

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW, get_cosine_lr


def get_random_batch(batch_size: int, vocab_size, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    batch_tokens = [torch.randint(vocab_size, (context_length + 1,), dtype=torch.long, device=device) for _ in range(batch_size)]

    x = torch.stack([tokens[:context_length] for tokens in batch_tokens])
    y = torch.stack([tokens[1:] for tokens in batch_tokens])
    return x, y


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_step(inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c, run_forward, run_backward, run_optimizer):
    optimizer.zero_grad()
    assert run_forward if run_backward else True

    forward_time = backward_time = optimizer_time = None

    if run_forward:
        sync()
        t0 = timeit.default_timer()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        sync()
        forward_time = timeit.default_timer() - t0

    if run_backward:
        sync()
        t0 = timeit.default_timer()
        loss.backward()
        sync()
        backward_time = timeit.default_timer() - t0

    if run_optimizer:
        sync()
        t0 = timeit.default_timer()
        clip_gradient(model.parameters(), max_norm)
        for g in optimizer.param_groups:
            g["lr"] = get_cosine_lr(lr_max, lr_min, t_w, t_c, step + 1)
        optimizer.step()
        sync()
        optimizer_time = timeit.default_timer() - t0

    return forward_time, backward_time, optimizer_time


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
    # benchmark
    batch_size: int = 4,
    warmup_steps: int,  # no warmup
    benchmark_steps: int,
    run_forward: bool,
    run_backward: bool,
    run_optimizer: bool,
    # graddient checkpoint
    gradient_checkpointing: bool,
    layer_chunk_size: int,
    # model
    vocab_size: int = 10000,
    context_length: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    rope_theta: float = 10000,
    # gradient clipping
    max_norm: float = 1.0,
    # optimizer
    lr_max: float = 1e-3,
    lr_min: float = 1e-4,
    t_w: int = 100,
    t_c: int = 10,
    # betas: tuple[float, float] = (0.9, 0.999),
    # weight_decay: float = 0.01,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dtype = get_dtype(mixed_precision)

    with torch.autocast(device, dtype=dtype):
        # vocab_size = get_vocab_size(vocab_path)  # NOTE: keep it now
        # print("vocab_size:", vocab_size)
        print("Init model...")
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

        print("Init optimizer...")
        optimizer = AdamW(model.parameters())  # use default values

        # reuse random inputs
        inputs, targets = get_random_batch(batch_size, vocab_size, context_length, device)

        torch.cuda.memory._record_memory_history(max_entries=1000000)

        print("Warmup...")
        for step in range(warmup_steps):
            run_step(inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c, run_forward, run_backward, run_optimizer)

        sync()

        # with torch.cuda.profiler.profile():

        print("Benchmarking...")

        # Start recording memory history.

        forward_times = []
        backward_times = []
        optimizer_times = []
        for step in range(benchmark_steps):
            ft, bt, ot = run_step(inputs, targets, step, model, optimizer, max_norm, lr_max, lr_min, t_w, t_c, run_forward, run_backward, run_optimizer)
            forward_times.append(ft)
            backward_times.append(bt)
            optimizer_times.append(ot)

        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot(f"/home/wilsonhong/gdrive/cs336/{mem_prof_file}.pickle")
        # Stop recording history.
        torch.cuda.memory._record_memory_history(enabled=None)

        if run_forward:
            print(f"[Forward] mean={statistics.mean(forward_times):.6f}, std={statistics.stdev(forward_times):.6f}")
        if run_backward:
            print(f"[Backard] mean={statistics.mean(backward_times):.6f}, std={statistics.stdev(backward_times):.6f}")
        if run_optimizer:
            print(f"[Optimizer] mean={statistics.mean(optimizer_times):.6f}, std={statistics.stdev(optimizer_times):.6f}")

        print("Done")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM")

    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--benchmark_steps", type=int, default=6)

    parser.add_argument("--run_forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_optimizer", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)

    parser.add_argument("--mem_prof_file", type=str, default="memory_profile.pickle")
    parser.add_argument("-mp", "--mixed_precision", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer_chunk_size", type=int, default=1)
    args = parser.parse_args()

    benchmark(**vars(args))


if __name__ == "__main__":
    main()
