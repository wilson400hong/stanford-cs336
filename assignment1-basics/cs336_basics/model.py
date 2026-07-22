import math

import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        # return x @ self.W.T


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
        )
        torch.nn.init.trunc_normal_(self.embeddings, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.G = torch.nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        denom = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        res = (x * self.G) / denom
        return res.to(in_dtype)
