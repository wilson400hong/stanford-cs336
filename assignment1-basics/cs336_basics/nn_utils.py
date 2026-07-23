import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    normalized = x - torch.amax(x, dim=dim, keepdim=True)
    exp = torch.exp(normalized)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy():
    pass
