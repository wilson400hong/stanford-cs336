from typing import Iterable

import torch


# def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
#     normalized = x - torch.amax(x, dim=dim, keepdim=True)
#     exp = torch.exp(normalized)
#     return exp / torch.sum(exp, dim=dim, keepdim=True)


def softmax(x: torch.Tensor, dim: int, temp: float | None = None) -> torch.Tensor:
    if temp == 0.0:
        # special case
        res = torch.zeros_like(x)
        res.scatter_(dim, x.argmax(dim=dim, keepdim=True), 1.0)
        return res
    normalized = x - torch.amax(x, dim=dim, keepdim=True)
    if temp is not None:
        normalized /= temp
    exp = torch.exp(normalized)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(logits, targets):
    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    shifted_logits = logits - max_logits

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    log_probs = shifted_logits - log_sum_exp

    # (2, 4) -> (2, 4, 1), gather requires same number of dimensions
    target_idx = targets.unsqueeze(-1)
    target_log_prob = torch.gather(dim=-1, index=target_idx, input=log_probs).squeeze(
        -1
    )

    ce = -target_log_prob.mean()
    return ce


@torch.no_grad
def gradient_clipping(
    params: Iterable[torch.Tensor], max_norm: float, eps: float = 1e-6
):
    params = [p for p in params if p.grad is not None]  # materialize
    total_norm = sum((p.grad**2).sum() for p in params).sqrt()
    if total_norm <= max_norm:
        return
    for p in params:
        p.grad.mul_(max_norm / (total_norm + eps))


def apply_top_p(probs: torch.Tensor, top_p: float = 0.9):
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cum_probs > top_p  # >= is precise, but > is safer
    # shift mask right. `clone()` is necessary for in-place shift
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # always keep the first

    # indices_to_remove[i] will be True if probs[i] should be removed
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
    indices_to_remove.scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    return probs.masked_fill(indices_to_remove, 0.0)
