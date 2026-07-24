import os

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


def train(
    # data
    dataset_path: str,
    # checkpoint
    checkpoint_dir: str,  # use as prefix
    checkpoint_interval: int,
    resume_checkpoint: bool,
    # training
    batch_size: int,
    steps: int,
    # model
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    # gradient clipping
    max_norm: float,
    # optimizer
    lr_max: float,  # 1e-4
    lr_min: float,
    t_w: int,
    t_c: int,
    betas: tuple[float, float],  # (0.9, 0.95)
    weight_decay: float,  # 0.01
    # logging
    logging_interval: int,
):
    # create checkpoint_dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        torch.device(device),
        torch.float32,
    )
    optimizer = AdamW(model.parameters(), lr_max, betas, weight_decay, eps=1e-8)
    step = 1
    if resume_checkpoint:
        load_path = get_latest_checkpoint_path(checkpoint_dir)
        if load_path:
            print(f"load checkpoint from {load_path}")
            step = load_checkpoint(load_path, model, optimizer) + 1
        else:
            print("no checkpoint found, start from scratch")

    while step <= steps:
        optimizer.zero_grad()
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        gradient_clipping(model.parameters(), max_norm, eps=1e-6)

        for g in optimizer.param_groups:
            g["lr"] = lr_cosine_schedule(lr_max, lr_min, t_w, t_c, step)

        optimizer.step()

        if step % logging_interval == 0:
            print(f"[{step}] loss: {loss.item()}")

        if step % checkpoint_interval == 0:
            save_path = f"{checkpoint_dir}/checkpoint_{step}.pt"
            print(f"[{step}] save checkpoint at {save_path}")
            save_checkpoint(model, optimizer, step, save_path)

        # TODO: eval... inside train loop or not?
        step += 1
