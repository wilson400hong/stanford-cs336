import math

import torch
from einops import einsum, rearrange, repeat


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
            torch.ones(d_model, dtype=dtype, device=device)
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


class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Adjacent rotation
    for [x0, x1, x2, x3]
    Rotate (x0, x1) and (x2, x3)
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._init_cache(max_seq_len, device)

    def _init_cache(self, max_seq_len: int, device: torch.device):
        self.max_seq_len = max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        # (max_seq_len, d_k/2)
        freqs = torch.outer(t, self.inv_freq)
        # freqs = torch.einsum("i, j -> i j", t, self.inv_freq)

        # [00, 01] -> [θ0, θ0, θ1, θ1]
        # (max_seq_len, d_k)
        emb = repeat(freqs, "s d -> s (d v)", v=2)

        self.register_buffer("cos_cached", emb.cos().to(device), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(device), persistent=False)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        """
        [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        """
        x_even, x_odd = rearrange(x, "... (d v) -> v ... d", v=2)
        x_rotated = torch.stack((-x_odd, x_even), dim=0)
        return rearrange(x_rotated, "v ... d -> ... (d v)")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # TODO: with multi-head attention, broadcast might be wrong.
        # # 1. extend cache
        max_pos = torch.max(token_positions).item()
        if max_pos >= self.max_seq_len:
            print("[INFO] Extending RoPE cache to", max_pos + 128)
            self._init_cache(max_pos + 128, device=x.device)

        # 2. advance index and unsqueeze to broadcast
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 3. apply RoPE
        return (x * cos) + (self.rotate_every_two(x) * sin)


# class ModernRotaryPositionalEmbedding(torch.nn.Module):
#     """
#     Half - Half rotation
#     for [x0, x1, x2, x3]
#     Rotate (x0, x2) and (x1, x3)
#     """

#     def __init__(
#         self,
#         theta: float,
#         d_k: int,
#         max_seq_len: int,
#         device: torch.device | None = None,
#     ):
#         super().__init__()
#         inv_freq = 1.0 / (
#             theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
#         )
#         self.register_buffer("inv_freq", inv_freq)
#         self._init_cache(max_seq_len, device)

#     def _init_cache(self, max_seq_len: int, device: torch.device):
#         self.max_seq_len = max_seq_len
#         t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
#         # (max_seq_len, d_k/2)
#         freqs = torch.outer(t, self.inv_freq)
#         # (max_seq_len, d_k)
#         emb = torch.cat((freqs, freqs), dim=-1)

#         self.register_buffer("cos_cached", emb.cos().to(device), persistent=False)
#         self.register_buffer("sin_cached", emb.sin().to(device), persistent=False)

#     def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         [x0, x1, x2, x3, x4, x5, x6, x7] -> [-x4, -x5, -x6, -x7, x0, x1, x2, x3]
#         """
#         x1 = x[..., : x.shape[-1] // 2]
#         x2 = x[..., x.shape[-1] // 2 :]
#         return torch.cat((-x2, x1), dim=-1)

#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
#         # # 1. extend cache
#         max_pos = torch.max(token_positions).item()
#         if max_pos >= self.max_seq_len:
#             self._init_cache(max_pos + 128, device=x.device)

#         # 2. Advanced Indexing and unsqueeze to broadcast
#         cos = self.cos_cached[token_positions].unsqueeze(1)
#         sin = self.sin_cached[token_positions].unsqueeze(1)
#         # 3. apply RoPE
#         return (x * cos) + (self.rotate_half(x) * sin)
