import triton
import triton.language as tl
import torch
from einops import rearrange
import timeit
import math

TILE_SIZE = 16


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        QB: (batch_size, n_queries, D)
        KB: (batch_size, n_keys, D)
        VB: (batch_size, n_keys, D)
        """
        device = Q.device
        B, N, D = Q=.shape  # N: seq_len


        O = torch.zeros_like(Q)
        L = torch.zeros((B, N), dtype=torch.float32, device=device)



        for b in range(B):
            Q_ = Q[b]
            K_ = K[b]
            V_ = V[b]

            O_ = O[b]
            L_ = L[b]


            for i in range(0, N, TILE_SIZE):
                Q_i = Q_[i:i+TILE_SIZE, :]
                O_i = O_[i:i+TILE_SIZE, :]
                L_i = L_[i:i+TILE_SIZE]

                l = torch.zeros((TILE_SIZE,), dtype=torch.float32, device=device)
                m = torch.full((TILE_SIZE,), float("-inf"), device=device)

                for j in range(0, N, TILE_SIZE):
                    K_j = K_[j:j+TILE_SIZE,:]
                    V_j = V_[j:j+TILE_SIZE,:]
                    S_ij = Q_i @ K_j / math.sqrt(D)
                    rowmax = torch.amax(S_ij, dim=-1, keepdim=True)
                    new_m =  torch.maximum(m, rowmax)
                    P_ij = torch.exp(S_ij - new_m)

                    delta_m = torch.exp(m - new_m)
                    new_l = l * delta_m + torch.sum(P_ij, dim=-1, keepdim=True)

                    O_i = torch.diag(delta_m) @ O_i + P_ij @ V_j

                    l = new_l
                    m = new_m

                O_i = torch.inverse(torch.diag(l)) @ O_i  # TODO
                L_i = m + torch.log(l)
                





    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("impelement me!")


# class WeightedSumFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight):
#         D, output_dims = x.shape[-1], x.shape[:-1]

#         input_shape = x.shape
#         x = rearrange(x, "... d -> (...) d")

#         ctx.save_for_backward(x, weight)

#         assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
#         assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
#         assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

#         ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # roughly 16 loops
#         ctx.ROWS_TILE_SIZE = 16
#         ctx.input_shape = input_shape

#         y = torch.empty(output_dims, device=x.device)

#         n_rows = y.numel()

#         weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
#             x,
#             weight,
#             y,
#             x.stride(0),
#             x.stride(1),
#             weight.stride(0),
#             y.stride(0),
#             NUM_ROWS=n_rows,
#             D=D,
#             ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
#             D_TILE_SIZE=ctx.D_TILE_SIZE,
#         )

#         return y.view(input_shape[:-1])

#     @staticmethod
#     def backward(ctx, grad_out):
#         x, weight = ctx.saved_tensors  # from ctx.save_for_backward(x, weight)
#         ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
#         n_rows, D = x.shape

#         partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
#         grad_x = torch.empty_like(x)

#         weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
#             x,
#             weight,
#             grad_out,
#             grad_x,
#             partial_grad_weight,
#             x.stride(0),
#             x.stride(1),
#             weight.stride(0),
#             grad_out.stride(0),
#             grad_x.stride(0),
#             grad_x.stride(1),
#             partial_grad_weight.stride(0),
#             partial_grad_weight.stride(1),
#             NUM_ROWS=n_rows,
#             D=D,
#             ROWS_TILE_SIZE=ROWS_TILE_SIZE,
#             D_TILE_SIZE=D_TILE_SIZE,
#         )

#         grad_weight = partial_grad_weight.sum(axis=0)
#         return grad_x, grad_weight


device = "cuda" if torch.cuda.is_available() else "cpu"

R = 16 * 1024
D = 8192
x = torch.rand((R, D), device=device, requires_grad=True)
w = torch.rand((D,), device=device, requires_grad=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# def check(shape):
#     torch.manual_seed(0)
#     x = torch.rand(*shape, device="cuda", requires_grad=True)
#     w = torch.rand(shape[-1], device="cuda", requires_grad=True)

#     F = WeightedSumFunc.apply

#     y = F(x, w)
#     assert y.grad_fn is not None, "no grad_fn — did inputs set requires_grad?"

#     # a scalar loss (use pow(2) so grads aren't all-ones/constant — a stronger test)
#     F(x, w).pow(2).sum().backward()

#     # reference from pure PyTorch autograd
#     xr = x.detach().clone().requires_grad_()
#     wr = w.detach().clone().requires_grad_()
#     (xr * wr).sum(-1).pow(2).sum().backward()

#     gx = torch.allclose(x.grad, xr.grad, atol=1e-2, rtol=1e-2)
#     gw = torch.allclose(w.grad, wr.grad, atol=1e-2, rtol=1e-2)
#     print(f"{str(shape):14s}  grad_x={gx}  grad_w={gw}")
#     assert gx and gw


# for shape in [(64, 128), (70, 300), (1024, 8192)]:  # incl. non-mult rows, non-pow2 D
#     check(shape)
