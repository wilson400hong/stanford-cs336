from torch.nn.modules import padding
from networkx.algorithms import boundary
from timeit import timeit
import triton
import triton.language as tl
import torch
from einops import rearrange
import timeit


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stide_row,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(
            NUM_ROWS,
            D,
        ),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(D,), strides=(weight_stride_dim,), offsets=(0,), block_shape=(D_TILE_SIZE,), order=(0,))

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stide_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),  # !!
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # D dim tiling
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        output += tl.sum(row * weight[None, :], axis=-1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # original inputs
    grad_output_ptr,  # grad input
    grad_x_ptr,
    partial_grad_weight_ptr,  # Grad outputs
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,  # ??
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)  # ??

    # grad inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # !!!!!!! this is 2D  (n_row_tils, D_TILE_SIZE ) !!!!!
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")  # (ROWS_TILE_SIZE, )

        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE, )
        grad_x_row = grad_output[:, None] * weight[None, :]  # (R,1) * (1,D) -> (R,D)
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # reduce grad_weight
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)  # (1, D_TILE_SIZE)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        # move pointer
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]

        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # roughly 16 loops
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device=x.device)

        n_rows = y.numel()

        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors  # from ctx.save_for_backward(x, weight)
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x,
            weight,
            grad_out,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight


device = "cuda" if torch.cuda.is_available() else "cpu"


R = 16 * 1024
D = 8192
x = torch.rand((R, D), device=device, requires_grad=True)
w = torch.rand((D,), device=device, requires_grad=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# sync()
# for _ in range(10):
#     y = (x * w).sum(-1)

# sync()
# t0 = timeit.default_timer()
# for _ in range(1000):
#     y = (x * w).sum(-1)

# sync()
# t1 = timeit.default_timer()
# print(f"Native elapsed: {t1 - t0:.6f}")


sync()
f_weightedsum = WeightedSumFunc.apply

# warmup
for _ in range(10):
    y = f_weightedsum(x, w)

sync()
t2 = timeit.default_timer()
for _ in range(1000):
    y = f_weightedsum(x, w)

sync()
t3 = timeit.default_timer()
print(f"Triton elapsed: {t3 - t2:.6f}")


print(x)
print(y)


def check(shape):
    torch.manual_seed(0)
    x = torch.rand(*shape, device="cuda", requires_grad=True)
    w = torch.rand(shape[-1], device="cuda", requires_grad=True)

    F = WeightedSumFunc.apply

    y = F(x, w)
    assert y.grad_fn is not None, "no grad_fn — did inputs set requires_grad?"

    # a scalar loss (use pow(2) so grads aren't all-ones/constant — a stronger test)
    F(x, w).pow(2).sum().backward()

    # reference from pure PyTorch autograd
    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    (xr * wr).sum(-1).pow(2).sum().backward()

    gx = torch.allclose(x.grad, xr.grad, atol=1e-2, rtol=1e-2)
    gw = torch.allclose(w.grad, wr.grad, atol=1e-2, rtol=1e-2)
    print(f"{str(shape):14s}  grad_x={gx}  grad_w={gw}")
    assert gx and gw


for shape in [(64, 128), (70, 300), (1024, 8192)]:  # incl. non-mult rows, non-pow2 D
    check(shape)
