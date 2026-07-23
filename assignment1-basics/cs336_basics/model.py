import math

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor


# TODO: replace 2-D weights with Linear. This might need fix load_state_dict. Not urgent now


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
        self.weight = torch.nn.Parameter(
            torch.empty(
                (out_features, in_features),  # since we use x @ w.T
                dtype=dtype,
                device=device,
            )
        )
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, self.weight, "... d_in, d_out d_in -> ... d_out"
        )  # x @ weight.T


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                dtype=dtype,
                device=device,
            )
        )
        torch.nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


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
        # NOTE: in_dtype may cause mixed precision bad
        return x_normed.to(in_dtype) * self.weight


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))


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
        freqs = torch.outer(
            t, self.inv_freq
        )  # freqs = torch.einsum("i, j -> i j", t, self.inv_freq)

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
        # NOTE: MHA should handle token_positions shape. Not RoPE's responsibility
        # NOTE: max(token_positions) should < max_pos
        # max_pos = torch.max(token_positions).item()
        # if max_pos >= self.max_seq_len:
        #     self._init_cache(max_pos + 128, device=x.device)

        # advance index and unsqueeze to broadcast
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # apply RoPE
        return (x * cos) + (self.rotate_every_two(x) * sin)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    normalized = x - torch.amax(x, dim=dim, keepdim=True)
    exp = torch.exp(normalized)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]

    scores = einsum(
        Q, K, "... queries dk, ... keys dk -> ... queries keys"
    ) / math.sqrt(d_k)
    if mask is not None:
        attn_mask = torch.where(mask, 0.0, float("-inf"))
        scores = scores + attn_mask
    scores = softmax(scores, dim=-1)
    return scores @ V


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float | None = None,  # RoPE
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)

        causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.rope = (
            RotaryPositionalEmbedding(theta, d_k, max_seq_len, device)
            if theta is not None
            else None
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        # q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # slice
        qh = rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        kh = rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        vh = rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        # RoPE
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            qh = self.rope(qh, token_positions)
            kh = self.rope(kh, token_positions)

        mask = self.causal_mask[:seq_len, :seq_len]

        attn = scaled_dot_product_attention(qh, kh, vh, mask)
        attn = rearrange(attn, "... h s d -> ... s (h d)")

        return self.o_proj(attn)


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
