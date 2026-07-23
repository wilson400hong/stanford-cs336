import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-8
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for params in group["params"]:
                if params.grad is None:
                    continue

                state = self.state[params]
                if len(state) == 0:
                    # initializaation
                    state["t"] = 1
                    state["m"] = torch.zeros_like(params)
                    state["v"] = torch.zeros_like(params)

                t = state["t"]
                m = state["m"]
                v = state["v"]

                grad = params.grad.data
                adjusted_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                params.data -= lr * weight_decay * params.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                params.data -= adjusted_lr * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
