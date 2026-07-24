import argparse
import os
import pickle
import time

import numpy as np
import torch

from .data import get_batch
from .model import TransformerLM
from .nn_utils import cross_entropy, gradient_clipping
from .optimizer import AdamW, lr_cosine_schedule
from .serialization import load_checkpoint, save_checkpoint


def get_latest_checkpoint_path(checkpoint_dir: str) -> str:
    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = [f for f in checkpoints if f.startswith("checkpoint_")]
    checkpoints = sorted(
        checkpoints, key=lambda x: int(os.path.splitext(x)[0].split("_")[1])
    )
    if len(checkpoints) == 0:
        return ""
    return os.path.join(checkpoint_dir, checkpoints[-1])


def save_checkpoint_file(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    overwrite: bool = False,
) -> str:
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{step}.pt"
    if os.path.exists(checkpoint_path) and (not overwrite):
        print(f"Checkpoint {checkpoint_path} exists. Skipped")
        return

    save_checkpoint(model, optimizer, step, checkpoint_path)
    return checkpoint_path


def get_vocab_size(vocab_path: str) -> int:
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return len(vocab)


@torch.no_grad()
def evaluate(
    valid_dataset: np.memmap,
    batch_size: int,
    eval_steps: int,
    context_length: int,
    model: TransformerLM,
    device: str,
) -> torch.Tensor:
    print("Evaluting...")
    model.eval()
    losses = torch.zeros(eval_steps, device=device)
    for i in range(eval_steps):
        inputs, targets = get_batch(valid_dataset, batch_size, context_length, device)
        losses[i] = cross_entropy(model(inputs), targets)

    model.train()
    return losses.mean()


# NOTE: defaults are for TinyStories
def train(
    *,
    # data
    train_dataset_path: str,
    eval_dataset_path: str,
    # training
    batch_size: int = 256,
    train_steps: int = 5000,  # batch_size * train_steps ~= 1_280_000
    # checkpoint
    checkpoint_dir: str,  # use as prefix
    checkpoint_interval: int = 1000,
    resume_checkpoint: bool = False,
    # eval
    eval_interval: int = 100,
    eval_steps: int = 20,
    # model
    vocab_size: int = 10000,  # TinyStories
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    rope_theta: float = 10000,
    # gradient clipping
    max_norm: float = 1.0,
    # optimizer
    lr_max: float,
    lr_min: float,
    t_w: int,
    t_c: int,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.01,
):
    # create checkpoint_dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Load dataset...")
    train_dataset = np.memmap(train_dataset_path, dtype=np.uint16, mode="r")
    eval_dataset = np.memmap(eval_dataset_path, dtype=np.uint16, mode="r")

    # vocab_size = get_vocab_size(vocab_path)  # NOTE: keep it now
    # print("vocab_size:", vocab_size)
    print("Init model...")
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        1e-5,
        torch.device(device),
        torch.float32,
    )
    print("Init optimizer...")
    optimizer = AdamW(model.parameters(), lr_max, betas, weight_decay, eps=1e-8)
    step = 1

    if resume_checkpoint:
        print("Load checkpoint...")

        load_path = get_latest_checkpoint_path(checkpoint_dir)
        if load_path:
            print(f"load checkpoint from {load_path}")
            step = load_checkpoint(load_path, model, optimizer) + 1
        else:
            print("no checkpoint found, start from scratch")

    print("Training...")
    t0 = time.perf_counter()
    processed = 0
    while step <= train_steps:
        optimizer.zero_grad()
        inputs, targets = get_batch(train_dataset, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        gradient_clipping(model.parameters(), max_norm, eps=1e-6)

        for g in optimizer.param_groups:
            g["lr"] = lr_cosine_schedule(lr_max, lr_min, t_w, t_c, step)

        optimizer.step()

        processed += batch_size * context_length

        if step % eval_interval == 0:
            elapsed = time.perf_counter() - t0
            eval_loss = evaluate(
                eval_dataset, batch_size, eval_steps, context_length, model, device
            )
            print(
                f"[{step}] Elapsed: {elapsed:.3f}  Train loss: {loss.item()}. Eval loss: {eval_loss.item()} "
            )
            print(f"[{step}] Train token/s: {processed / elapsed}")

        if step % checkpoint_interval == 0:
            save_path = save_checkpoint_file(model, optimizer, step, checkpoint_dir)
            print(f"[{step}] Saved checkpoint: {save_path}")

        step += 1

    save_path = save_checkpoint_file(model, optimizer, step - 1, checkpoint_dir)
    print(f"[{step}] Saved final checkpoint: {save_path}")

    print("Done")


def main():
    parser = argparse.ArgumentParser(description="Train TransformerLM")

    # data — required
    parser.add_argument("-t", "--train_dataset_path", type=str, required=True)
    parser.add_argument("-e", "--eval_dataset_path", type=str, required=True)

    # training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=5000)

    # checkpoint
    parser.add_argument("-o", "--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument(
        "--resume_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resume from latest checkpoint if exists (use --no-resume_checkpoint to disable)",
    )

    # eval
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=20)

    # model
    parser.add_argument(
        "--vocab_size", type=int, default=10000, help="TinyStories=10000"
    )
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)

    # gradient clipping
    parser.add_argument("--max_norm", type=float, default=1.0)

    # optimizer
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)

    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.9, 0.95),
        help="AdamW betas, e.g. --betas 0.9 0.95",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()
    # argparse gives list for nargs, train() expects tuple
    args.betas = tuple(args.betas)
    args.t_c = args.train_steps
    args.t_w = 0.03 * args.train_steps

    train(**vars(args))


if __name__ == "__main__":
    main()
