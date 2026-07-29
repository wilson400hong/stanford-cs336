import statistics
import argparse

import torch
import torch.cuda.nvtx as nvtx


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 10, bias=False)
        self.ln = torch.nn.LayerNorm(10)
        self.fc2 = torch.nn.Linear(10, out_features, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        print("inside forward")
        fc1 = self.fc1(x)
        print(f"{fc1.dtype=}")

        relu = self.relu(fc1)
        print(f"{relu.dtype=}")

        ln = self.ln(relu)
        print(f"{ln.dtype=}")

        fc2 = self.fc2(ln)
        print(f"{fc2.dtype=}")
        return fc2


def print_dtype(module):
    for name, param in module.named_parameters():
        print(name, param.dtype, param.grad.dtype if param.grad is not None else None)


def benchmark(
    warmup_steps: int,
    benchmark_steps: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f32_toy = ToyModel(in_features=5, out_features=4).to(device)

    print("Original:")
    print_dtype(f32_toy)

    with torch.autocast(device, dtype=torch.bfloat16):
        f16_toy = ToyModel(in_features=5, out_features=4).to(device)
        print("Autocast")
        print_dtype(f16_toy)

        x = torch.rand((3, 5), dtype=torch.float32, device=device)

        print(f"{x.dtype=}")

        y = f16_toy(x)

        print_dtype(f16_toy)

        print(f"{y.dtype=}")

        loss = torch.exp(y).sum()
        print(f"{loss.dtype=}")

        loss.backward()

        print_dtype(f16_toy)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Toy")

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=3)

    args = parser.parse_args()
    # argparse gives list for nargs, train() expects tuple

    benchmark(**vars(args))


if __name__ == "__main__":
    main()


"""
uv run python -m cs336_systems.benchmark_toy
"""


"""
torch.autocast(dtype=torch.float16):

• the model parameters within the autocast context -> fp32
• the output of the first feed-forward layer (ToyModel.fc1) -> fp16
• the output of layer norm (ToyModel.ln) -> fp32
• the model’s predicted logits -> fp16
• the loss -> fp32
• the model’s gradients -> fp32


torch.autocast(dtype=torch.bf16):

• the model parameters within the autocast context ->
• the output of the first feed-forward layer (ToyModel.fc1) -> 
• the output of layer norm (ToyModel.ln) -> 
• the model’s predicted logits -> 
• the loss -> 
• the model’s gradients -> 



"""
