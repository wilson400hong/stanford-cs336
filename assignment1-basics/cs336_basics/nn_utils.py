from typing import Iterable

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    normalized = x - torch.amax(x, dim=dim, keepdim=True)
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
