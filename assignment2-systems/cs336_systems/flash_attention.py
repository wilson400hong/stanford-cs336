import triton
import triton.language as tl
import torch
import math
from einops import einsum, rearrange
import timeit
import statistics

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
        assert seq_len % T == 0, "seq_len must be divisible by T"

        sqrt_d = math.sqrt(d_model)

        O_tiles = []
        L_tiles = []

        # outer loop: Query tiling
        for i in range(0, seq_len, T):
            # Q_i shape: (B, T, D)
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
        L = torch.cat(L_tiles, dim=1).squeeze(-1)  # (B, N)

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
                L_i = L[:, i : i + T].unsqueeze(-1)
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
            L_i = L[:, i : i + T].unsqueeze(-1)
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


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,  # output
    L_ptr,  # log sum exp
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,  # 1 / sqrt(D)
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    i = tl.program_id(0)  # query_tile_index
    batch_idx = tl.program_id(1).to(tl.int64)  # batch axis

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + (batch_idx) * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + (batch_idx) * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # watch out!
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + (batch_idx) * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + (batch_idx) * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)  # (T, 1)
    m_i = tl.full((Q_TILE_SIZE, 1), float("-inf"), dtype=tl.float32)  # (T, 1)

    if is_causal:
        max_q = i * Q_TILE_SIZE + (Q_TILE_SIZE - 1)
        n_kv = tl.cdiv(max_q + 1, K_TILE_SIZE)
    else:
        n_kv = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(n_kv):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = tl.where(causal_mask, S_ij, float("-inf"))

        rowmax = tl.max(S_ij, axis=-1, keep_dims=True)  # (T, 1)
        new_m_i = tl.maximum(m_i, rowmax)  # (T, 1)

        P_ij = tl.exp(S_ij - new_m_i)
        delta_m_i = tl.exp(m_i - new_m_i)

        O_i = delta_m_i * O_i + tl.dot(P_ij, V_j)
        l_i = l_i * delta_m_i + tl.sum(P_ij, axis=-1, keep_dims=True)  # (T, 1)
        m_i = new_m_i

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i /= l_i

    L_i = tl.reshape(m_i + tl.log(l_i), (Q_TILE_SIZE,))

    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


@triton.jit
def flash_bwd_kernel_preproc(
    O_ptr,  # (N, D)
    dO_ptr,
    D_ptr,  # (N, 1)
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_db,
    stride_dq,
    N_QUERIES,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    """
    compute torch.sum(O * dO, dim=-1, keepdim=True)
    """
    i = tl.program_id(0)
    batch_idx = tl.program_id(1).to(tl.int64)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + (batch_idx) * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + (batch_idx) * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + (batch_idx) * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    D_i = tl.sum(O_i * dO_i, axis=1)

    tl.store(D_block_ptr, D_i, boundary_check=(0,))


@triton.jit
def flash_bwd_kernel_dkdv(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,  # log sum exp
    D_ptr,  # rowsum(O dot dO)
    dO_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale,  # 1 / sqrt(D)
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    j = tl.program_id(0)  # key_tile_index
    batch_idx = tl.program_id(1).to(tl.int64)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + (batch_idx) * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + (batch_idx) * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + (batch_idx) * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + (batch_idx) * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + (batch_idx) * stride_dkb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dkk, stride_dkd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + (batch_idx) * stride_dvb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dvk, stride_dvd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    dK_j = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)

    for i in range(0, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_i = tl.reshape(tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero"), (Q_TILE_SIZE, 1))  # (T,)
        D_i = tl.reshape(tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero"), (Q_TILE_SIZE, 1))

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = tl.where(causal_mask, S_ij, float("-inf"))

        P_ij = tl.exp(S_ij - L_i)

        dV_j += tl.dot(tl.trans(P_ij), dO_i)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i)
        dK_j += scale * tl.dot(tl.trans(dS_ij), Q_i)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dV_block_ptr, dV_j, boundary_check=(0, 1))
    tl.store(dK_block_ptr, dK_j, boundary_check=(0, 1))


@triton.jit
def flash_bwd_kernel_dq(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,  # log sum exp
    D_ptr,  # rowsum(O dot dO)
    dO_ptr,
    dQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    N_QUERIES,
    N_KEYS,
    scale,  # 1 / sqrt(D)
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    i = tl.program_id(0)  # query_tile_index
    batch_idx = tl.program_id(1).to(tl.int64)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + (batch_idx) * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + (batch_idx) * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + (batch_idx) * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + (batch_idx) * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + (batch_idx) * stride_dqb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_dqq, stride_dqd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L_i = tl.reshape(tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero"), (Q_TILE_SIZE, 1))  # (T,)
    D_i = tl.reshape(tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero"), (Q_TILE_SIZE, 1))

    dQ_i = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)

    if is_causal:
        max_q = i * Q_TILE_SIZE + (Q_TILE_SIZE - 1)
        n_kv = tl.cdiv(max_q + 1, K_TILE_SIZE)
    else:
        n_kv = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(n_kv):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S_ij = tl.where(causal_mask, S_ij, float("-inf"))

        P_ij = tl.exp(S_ij - L_i)

        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i)

        dQ_i += scale * tl.dot(dS_ij, K_j)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_i, boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal, tile_size=TILE_SIZE):
        B, N, D_MODEL = Q.shape
        device = Q.device
        scale = 1.0 / math.sqrt(D_MODEL)
        O = torch.empty((B, N, D_MODEL), dtype=torch.float32, device=device)
        L = torch.empty(
            (B, N),
            dtype=torch.float32,
            device=device,
        )
        ctx.is_causal = is_causal
        ctx.tile_size = tile_size
        flash_fwd_kernel[(triton.cdiv(N, ctx.tile_size), B)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N,
            N,
            scale,
            D_MODEL,
            ctx.tile_size,
            ctx.tile_size,
            ctx.is_causal,
            num_stages=1,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        B, N, D_MODEL = Q.shape
        device = Q.device
        scale = 1.0 / math.sqrt(D_MODEL)

        D = torch.empty((B, N), dtype=torch.float32, device=device)

        flash_bwd_kernel_preproc[(triton.cdiv(N, ctx.tile_size), B)](
            O,
            dO,
            D,
            O.stride(0),
            O.stride(1),
            O.stride(2),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            D.stride(0),
            D.stride(1),
            N,
            D_MODEL,
            ctx.tile_size,
            num_stages=1,
        )

        dK = torch.empty((B, N, D_MODEL), dtype=torch.float32, device=device)
        dV = torch.empty((B, N, D_MODEL), dtype=torch.float32, device=device)

        flash_bwd_kernel_dkdv[(triton.cdiv(N, ctx.tile_size), B)](
            Q,
            K,
            V,
            L,
            D,
            dO,
            dK,
            dV,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            N,
            N,
            scale,
            D_MODEL,
            ctx.tile_size,
            ctx.tile_size,
            ctx.is_causal,
            num_stages=1,
        )

        dQ = torch.empty((B, N, D_MODEL), dtype=torch.float32, device=device)

        flash_bwd_kernel_dq[(triton.cdiv(N, ctx.tile_size), B)](
            Q,
            K,
            V,
            L,
            D,
            dO,
            dQ,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            N,
            N,
            scale,
            D_MODEL,
            ctx.tile_size,
            ctx.tile_size,
            ctx.is_causal,
            num_stages=1,
        )

        return dQ, dK, dV, None, None


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


def check_triton(shape, is_causal, tile_size=TILE_SIZE):
    print(f"Check Triton FA : {shape=} {is_causal=} {tile_size=}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B, N, Dm = shape

    Q = torch.rand(B, N, Dm, device=device, requires_grad=True)
    K = torch.rand(B, N, Dm, device=device, requires_grad=True)
    V = torch.rand(B, N, Dm, device=device, requires_grad=True)
    dO = torch.randn(B, N, Dm, device=device)

    # flash
    Of = FlashAttentionTriton.apply(Q, K, V, is_causal, tile_size)
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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


WARMUP_STEPS = 5
BENCH_STEPS = 50


def benchmark_attention(shape, func, warmup_steps=WARMUP_STEPS, bench_steps=BENCH_STEPS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B, N, Dm = shape
    is_causal = True

    # Reference - PyTorch

    for _ in range(WARMUP_STEPS):
        Q = torch.rand(B, N, Dm, device=device, requires_grad=True)
        K = torch.rand(B, N, Dm, device=device, requires_grad=True)
        V = torch.rand(B, N, Dm, device=device, requires_grad=True)
        dO = torch.randn(B, N, Dm, device=device)
        O = func(Q, K, V, is_causal)
        O.backward(dO)

    fwd_times = []
    bwd_times = []

    sync()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # <-- reset BEFORE the region

    peak = 0
    try:
        for _ in range(BENCH_STEPS):
            Q = torch.rand(B, N, Dm, device=device, requires_grad=True)
            K = torch.rand(B, N, Dm, device=device, requires_grad=True)
            V = torch.rand(B, N, Dm, device=device, requires_grad=True)
            dO = torch.randn(B, N, Dm, device=device)
            sync()

            t0 = timeit.default_timer()
            O = func(Q, K, V, is_causal)
            # O = sdpa(Qr, Kr, Vr, is_causal)
            sync()
            t1 = timeit.default_timer()
            fwd_times.append(t1 - t0)

            O.backward(dO)
            sync()
            t2 = timeit.default_timer()
            bwd_times.append(t2 - t1)
            peak = max(peak, torch.cuda.max_memory_allocated())
    except torch.cuda.OutOfMemoryError:
        peak = float("nan")

    torch.cuda.empty_cache()
    return (
        statistics.mean(fwd_times),
        statistics.mean(bwd_times),
        peak,
    )


def benchmark():
    B = 1
    print(f"{TILE_SIZE=}")
    for Dm in [256, 512]:
        for N in [256, 1024, 4096, 8192, 16384, 32768]:
            pytorch_fwd, pytorch_bwd, pytorch_mem = benchmark_attention((B, N, Dm), sdpa)
            triton_fwd, triton_bwd, triton_mem = benchmark_attention((B, N, Dm), FlashAttentionTriton.apply)
            print(f"[{B=} {N=} {Dm=}] {pytorch_fwd=:.4f} {pytorch_bwd=:.4f} {pytorch_mem=}    {triton_fwd=:.4f} {triton_bwd=:.4f} {triton_mem:.4f}")


def check():
    shape = [4, 1024, 256]
    for causal in [False, True]:
        print(f"# causal={causal}")
        print("##  check PyTorch impl...")
        check_pytorch(shape, causal)

        print("## check Triton impl...")
        check_triton(shape, causal)


if __name__ == "__main__":
    benchmark()
