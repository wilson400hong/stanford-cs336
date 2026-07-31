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


device = "cuda" if torch.cuda.is_available() else "cpu"


R = 16 * 1024
D = 8192
x = torch.rand((R, D), device=device)
w = torch.rand((D,), device=device)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


sync()
for _ in range(10):
    y = (x * w).sum(-1)

sync()
t0 = timeit.default_timer()
for _ in range(1000):
    y = (x * w).sum(-1)

sync()
t1 = timeit.default_timer()
print(f"Native elapsed: {t1 - t0:.6f}")


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
