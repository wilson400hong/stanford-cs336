import triton
import triton.language as tl
import torch
import timeit
import math
from einops import einsum, rearrange


TILE_SIZE = 16


def softmax(x, dim=-1):
    rescaled = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return rescaled / torch.sum(rescaled, dim=dim, keepdim=True)


def sdpa(Q, K, V, is_causal: bool):
    """Scaled dot-product attention"""
    _, seq_len, d_model = Q.shape
    device = Q.device
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_model)

    # Construct causal mask
    if is_causal:
        iota = torch.arange(seq_len, device=device)
        qi = rearrange(iota, "query -> query 1")
        kj = rearrange(iota, "key   -> 1   key")
        causal_mask = qi >= kj  # (query, key)
        attention_scores = torch.where(causal_mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal, T=TILE_SIZE):
        """
        Args:
            Q: (B, seq_len, D)
            K: (B, seq_len, D)
            V: (B, seq_len, D)

        Returns:
            O: (B, seq_len, D)
        """
        device = Q.device
        B, seq_len, d_model = Q.shape
        assert seq_len % T == 0, "seq_len must be divisible by tile_size"

        sqrt_d = math.sqrt(d_model)

        O_tiles = []
        L_tiles = []

        # outer loop: Query tiling
        for i in range(0, seq_len, T):
            # Q_i shape: (B, tile_size, D)
            Q_i = Q[:, i : i + T, :]

            # each tile has B in first dim
            O_i = torch.zeros((B, T, d_model), dtype=torch.float32, device=device)
            l_i = torch.zeros((B, T, 1), dtype=torch.float32, device=device)
            m_i = torch.full((B, T, 1), float("-inf"), dtype=torch.float32, device=device)

            # inner loop: Key, Value tiling
            for j in range(0, seq_len, T):
                # Causal: skip when j > i
                if is_causal and j > i:
                    break

                # K_j, V_j shape: (B, T, D)
                K_j = K[:, j : j + T, :]
                V_j = V[:, j : j + T, :]

                # Batched Matmul: (B, T, D) @ (B, D, T) -> (B, T, T)
                # use .mT to transpose shape[-1], shape[-2]
                S_ij = Q_i @ K_j.mT / sqrt_d

                # handle Causal Mask only when i == j
                if is_causal and i == j:
                    Q_mask = torch.arange(i, i + T, device=device)[:, None]
                    K_mask = torch.arange(j, j + T, device=device)[None, :]
                    causal_mask = Q_mask >= K_mask
                    # NOTE: we use -inf instead of -1e6 now
                    S_ij = torch.where(causal_mask, S_ij, float("-inf"))

                rowmax = torch.amax(S_ij, dim=-1, keepdim=True)
                new_m_i = torch.maximum(m_i, rowmax)

                P_ij = torch.exp(S_ij - new_m_i)
                delta_m_i = torch.exp(m_i - new_m_i)

                O_i = delta_m_i * O_i + P_ij @ V_j
                l_i = l_i * delta_m_i + torch.sum(P_ij, dim=-1, keepdim=True)
                m_i = new_m_i

            # normalization
            O_i /= l_i
            O_tiles.append(O_i)

            #  Log-Sum-Exp (L) for Backward
            L_i = m_i + torch.log(l_i)
            L_tiles.append(L_i)

        O = torch.cat(O_tiles, dim=1)  # (B, N, D)
        L = torch.cat(L_tiles, dim=1)  # (B, N, 1)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.tile_size = T

        return O

    @staticmethod
    def backward(ctx, grad_out):
        dO = grad_out
        L, Q, K, V, O = ctx.saved_tensors
        B, seq_len, d_model = Q.shape
        device = Q.device
        sqrt_d = math.sqrt(d_model)
        is_causal = ctx.is_causal
        T = ctx.tile_size

        # TODO: is_causal!!!

        D = torch.sum(O * dO, dim=-1, keepdim=True)

        dK_tiles = []
        dV_tiles = []

        for j in range(0, seq_len, T):
            K_j = K[:, j : j + T, :]
            V_j = V[:, j : j + T, :]

            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)

            for i in range(0, seq_len, T):
                if is_causal and j > i:
                    continue

                Q_i = Q[:, i : i + T, :]
                D_i = D[:, i : i + T, :]
                dO_i = dO[:, i : i + T, :]
                L_i = L[:, i : i + T, :]
                dO_i = dO[:, i : i + T, :]
                S_ij = Q_i @ K_j.mT / sqrt_d

                if is_causal and i == j:
                    Q_mask = torch.arange(i, i + T, device=device)[:, None]
                    K_mask = torch.arange(j, j + T, device=device)[None, :]
                    causal_mask = Q_mask >= K_mask
                    # NOTE: we use -inf instead of -1e6 now
                    S_ij = torch.where(causal_mask, S_ij, float("-inf"))

                P_ij = torch.exp(S_ij - L_i)

                dV_j += P_ij.mT @ dO_i
                dP_ij = dO_i @ V_j.mT

                dS_ij = P_ij * (dP_ij - D_i)
                dK_j += (dS_ij.mT @ Q_i) / sqrt_d

            dK_tiles.append(dK_j)
            dV_tiles.append(dV_j)

        dQ_tiles = []

        for i in range(0, seq_len, T):
            Q_i = Q[:, i : i + T, :]
            dO_i = dO[:, i : i + T, :]
            L_i = L[:, i : i + T, :]
            D_i = D[:, i : i + T, :]

            dQ_i = torch.zeros_like(Q_i)

            for j in range(0, seq_len, T):
                if is_causal and j > i:
                    break
                K_j = K[:, j : j + T, :]
                V_j = V[:, j : j + T, :]
                S_ij = Q_i @ K_j.mT / sqrt_d

                if is_causal and i == j:
                    Q_mask = torch.arange(i, i + T, device=device)[:, None]
                    K_mask = torch.arange(j, j + T, device=device)[None, :]
                    causal_mask = Q_mask >= K_mask
                    # NOTE: we use -inf instead of -1e6 now
                    S_ij = torch.where(causal_mask, S_ij, float("-inf"))

                P_ij = torch.exp(S_ij - L_i)
                dP_ij = dO_i @ V_j.mT
                dS_ij = P_ij * (dP_ij - D_i)
                dQ_i += (dS_ij @ K_j) / sqrt_d

            dQ_tiles.append(dQ_i)

        dQ = torch.cat(dQ_tiles, dim=1)
        dK = torch.cat(dK_tiles, dim=1)
        dV = torch.cat(dV_tiles, dim=1)

        return dQ, dK, dV, None, None


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def check_pytorch(shape, is_causal, T=TILE_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B, N, Dm = shape

    Q = torch.rand(B, N, Dm, device=device, requires_grad=True)
    K = torch.rand(B, N, Dm, device=device, requires_grad=True)
    V = torch.rand(B, N, Dm, device=device, requires_grad=True)
    dO = torch.randn(B, N, Dm, device=device)

    # flash
    Of = FlashAttentionPytorch.apply(Q, K, V, is_causal, T)
    Of.backward(dO)
    dQf, dKf, dVf = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    # reference: same inputs, fresh leaves, SAME dO
    Qr, Kr, Vr = (t.detach().clone().requires_grad_() for t in (Q, K, V))
    Or = sdpa(Qr, Kr, Vr, is_causal)
    Or.backward(dO)

    print(f"verify forward: {torch.allclose(Of, Or, atol=1e-2, rtol=1e-2)}")

    print("verify backward:")
    for name, a, b in [("dQ", dQf, Qr.grad), ("dK", dKf, Kr.grad), ("dV", dVf, Vr.grad)]:
        print(f"  {name}: {torch.allclose(a, b, atol=1e-2, rtol=1e-2)} maxerr={(a - b).abs().max():.2e}")


if __name__ == "__main__":
    for causal in [False, True]:
        print(f"### causal={causal}")
        check_pytorch([2, 256, 64], causal)


# # reference from pure PyTorch autograd
# xr = x.detach().clone().requires_grad_()
# wr = w.detach().clone().requires_grad_()
# (xr * wr).sum(-1).pow(2).sum().backward()

# gx = torch.allclose(x.grad, xr.grad, atol=1e-2, rtol=1e-2)
# gw = torch.allclose(w.grad, wr.grad, atol=1e-2, rtol=1e-2)
# print(f"{str(shape):14s}  grad_x={gx}  grad_w={gw}")
# assert gx and gw


# for shape in [(64, 128), (70, 300), (1024, 8192)]:  # incl. non-mult rows, non-pow2 D
#     check(shape)
