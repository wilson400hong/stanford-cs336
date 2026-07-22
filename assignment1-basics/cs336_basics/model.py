import math

import torch
from einops import einsum

# TODO: ensure requires_grad


def init_linear_weights(
    in_dim: int,
    out_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    w = torch.empty(
        (out_dim, in_dim),  # since we use x @ w.T
        dtype=dtype,
        device=device,
    )

    var = 2 / (in_dim + out_dim)
    std = math.sqrt(var)
    torch.nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)
    return w


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
            init_linear_weights(in_features, out_features, device, dtype)
        )

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
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed.to(in_dtype) * self.weight


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W1 = torch.nn.Parameter(init_linear_weights(d_model, d_ff, device, dtype))
        self.W2 = torch.nn.Parameter(init_linear_weights(d_ff, d_model, device, dtype))
        self.W3 = torch.nn.Parameter(init_linear_weights(d_model, d_ff, device, dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x @ self.W1.T
        silu = a * torch.sigmoid(a)
        b = x @ self.W3.T
        c = silu * b
        return c @ self.W2.T
