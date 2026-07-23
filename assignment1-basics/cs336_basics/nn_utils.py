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

    # (2, 4) -> (2, 4, 1)
    target_idx = targets.unsqueeze(-1)
    target_log_prob = torch.gather(dim=-1, index=target_idx, input=log_probs).squeeze(
        -1
    )

    ce = -target_log_prob.mean()
    return ce
