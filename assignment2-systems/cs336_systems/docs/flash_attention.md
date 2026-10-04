# FlashAttention in Triton — Learnings

Notes from implementing FlashAttention (forward + backward, PyTorch + Triton, causal + non-causal). All variants pass `test_flash_backward_{pytorch,triton}`.

## What was built

| Piece | Role |
|---|---|
| `sdpa` (naive) | reference oracle |
| `FlashAttentionPytorch` (fwd+bwd) | correctness oracle for the Triton version |
| `flash_fwd_kernel` | Triton forward (online softmax, stores `O` and `L`) |
| `flash_bwd_kernel_preproc` | fused `D = rowsum(O ⊙ dO)` |
| `flash_bwd_kernel_dkdv` | `dK`, `dV` — grid over **key** tiles, reduce over queries |
| `flash_bwd_kernel_dq` | `dQ` — grid over **query** tiles, reduce over keys |

## The core parallelization principle

The single idea that drives every kernel's structure:

> **Each program owns one disjoint output tile and loops over the reduction dimension.**
> The PyTorch **outer loop becomes the grid**; only the **inner (reduction) loop** stays in the kernel.

Consequences:
- **Forward:** grid over query tiles. `Q`/`O`/`L` are fixed per program (load/store once, never `.advance`); `K`/`V` tile the reduction (key) dim → `.advance` each iteration.
- **Backward needs two kernels** because the two gradients reduce over *different* axes:
  - `dK`/`dV` reduce over **queries** → grid over key tiles (each key tile owns its `dK_j`/`dV_j`).
  - `dQ` reduces over **keys** → grid over query tiles.
  - Splitting this way keeps each output tile disjoint per program → **no atomics**. A single kernel over one axis would force `tl.atomic_add` for the other gradient.
- `D = rowsum(O⊙dO)` is a shared per-query-row precompute → its own kernel (or PyTorch), so both backward kernels just read it instead of recomputing.

## Triton API gotchas (things that actually broke)

- **`@triton.jit` + `[grid]`** — every kernel needs the decorator, and every launch needs `kernel[grid](...)`. Calling a jit'd fn plainly → `Cannot call @triton.jit'd outside of the scope of a kernel`; a missing decorator → `'function' object is not subscriptable` on `[grid]`.
- **Reductions use `axis=`, not `dim=`** — `tl.max(x, axis=-1, keep_dims=True)`. `dim=` is a torch-ism → compile error.
- **No `.mT`** on Triton tensors — use `tl.trans(x)`.
- **No `break`** in a jit loop (`unsupported AST node type: Break`) — bound the loop range instead (`for j in range(n_kv)`).
- **`tl.arange(start, end)` needs constexpr bounds** — build indices as `runtime_offset + tl.arange(0, TILE)`, not `tl.arange(offset, offset+TILE)`.
- **`tl.sqrt` rejects int** — `sqrt_d = tl.sqrt(D)` fails when `D` is an int constexpr; use `D ** 0.5`.
- **`.advance(offsets)` returns a NEW block_ptr and needs its offset arg** — must reassign: `p = p.advance((TILE, 0))`. `p.advance()` (no arg) or discarding the result is a silent no-op → reads the same tile forever.
- **1-tuples need the trailing comma** — `order=(0,)`, `strides=(s,)`. `(0)` is just an int.
- **`tl.dot` uses TF32 by default** for fp32 → ~1e-2 error. Verify with `atol≈1e-2`, or pass `input_precision="ieee"`.
- **Cast batch offset to int64** — `base + batch_idx.to(tl.int64) * stride_b` (avoids int32 overflow for large tensors).

## Block pointers

- **`shape` = the whole (base-adjusted) tensor; `block_shape` = one tile.** `offsets` positions the tile; `.advance` slides it; `boundary_check` uses `shape` to zero-pad partial edge tiles.
- **Batch goes in the base pointer**, not the block_ptr: `Q_ptr + batch_idx*stride_qb`, then a **2-D** `(N, D)` block_ptr. Rank of the block_ptr = dims the tile spans *after* the base advance (so 2-D, even though the tensor is 3-D `(B,N,D)`).
- **`block_shape` entries must be constexpr** — this is why `D` (head dim) is passed as `D: tl.constexpr`.

## FlashAttention specifics

- **Online softmax, 2-D stats convention:** keep `m_i`, `l_i` as `(Q_TILE, 1)` (via `keep_dims=True`) so they broadcast against both `(Q_TILE, K_TILE)` scores and `(Q_TILE, D)` output — **no `[:, None]` needed anywhere**. (The 1-D alternative works but scatters `[:, None]` across ~4 sites.)
- **`L` (logsumexp) is stored `(B, N)`** — a per-query scalar. The test explicitly checks for a saved tensor of shape `(B, N)`. In the PyTorch impl, `squeeze(-1)` before saving and `unsqueeze(-1)` at use-site in the backward.
- **Backward saves only `L`** (not `P`) and **recomputes** `S`, `P = exp(S − L_i)` — the whole point of the memory savings.
- **Backward gradient math:** `dV += Pᵀ·dO`, `dP = dO·Vᵀ`, `dS = P ⊙ (dP − D_i)`, `dK += scale·dSᵀ·Q`, `dQ += scale·dS·K`. `D_i = rowsum(O⊙dO)` is the softmax-Jacobian correction term.

## Causal masking

- **Compare absolute positions:** `q_idx = tile_i*Q_TILE + arange(0,Q_TILE)`, `k_idx = tile_j*K_TILE + arange(0,K_TILE)`, `mask = q_idx[:,None] >= k_idx[None,:]`. Use **`>=`** (a token attends to itself — `>` wrongly masks the diagonal).
- **Different `Q_TILE_SIZE` vs `K_TILE_SIZE` breaks shortcuts:**
  - The `j == tile_idx` "diagonal only" mask is **wrong** for unequal tiles — the boundary straddles a different key tile. Fix: **mask every processed tile** when causal (fully-valid tiles get an all-True no-op mask).
  - The loop-skip bound must use positions, not tile indices: forward `n_kv = cdiv(query_start + Q_TILE, K_TILE)`.
- **The causal skip direction flips per kernel** (prefix vs suffix of valid tiles):
  - dQ / forward: loop keys `j ≤ i` → valid tiles are a **prefix** → bound the range / `break`-equivalent.
  - dK/dV: loop queries `i ≥ j` → valid tiles are a **suffix** → skip the prefix (`continue`, or start offset at `s_q = (j*K_TILE)//Q_TILE`). A wrong `s_q` (e.g. `cdiv(min_k-1, Q_TILE)`) silently drops gradient when `Q_TILE > K_TILE`.
  - Rule: skipping too *few* tiles is safe (mask zeros them); skipping too *many* is a silent correctness bug.

## Debugging methodology that worked

- **Oracle chain:** naive `sdpa` → PyTorch flash → Triton flash. Each layer verifies the next with `torch.allclose`.
- **Use random `dO`** (not `.sum()` / all-ones upstream) — a constant upstream grad can hide a wrong `dP` broadcast.
- **Test the cases that break assumptions:** `Q_TILE != K_TILE` (both directions), `N` not a multiple of the tile (boundary_check path), and **both** causal settings. These caught the diagonal-shortcut and `s_q` bugs that equal-tile tests missed.
- **Verify each kernel in isolation** before wiring the `autograd.Function` — launch the raw kernel with reference `L`/`D` and compare to autograd grads.
- **`autograd.Function` plumbing:** `@staticmethod` on both `forward`/`backward`; `save_for_backward` the exact tensors the test expects (and the right ones — `O` not a duplicate `Q`); return one grad per forward input (`None` for non-tensors like `is_causal`, `tile_size`).

## Benchmark: Triton flash vs naive PyTorch attention

Setup: `B=1`, `TILE_SIZE=16`, causal, forward + backward timed separately (warm-up 5, mean of 50), peak = `max_memory_allocated`. Baseline is the **naive** `sdpa` (materialized scores), *not* `F.scaled_dot_product_attention`.

| Dm | N | fwd pt | fwd tri | fwd× | bwd pt | bwd tri | bwd× | mem pt | mem tri | mem× |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4096 | 0.7ms | 0.4ms | 1.8× | 1.2ms | 1.5ms | 0.8× | 0.46 GB | 0.07 GB | 6.9× |
| 256 | 8192 | 2.3ms | 1.0ms | 2.3× | 4.2ms | 4.9ms | 0.9× | 1.75 GB | 0.12 GB | 14.8× |
| 256 | 16384 | 8.4ms | 3.2ms | 2.6× | 15.6ms | 18.5ms | 0.8× | 6.83 GB | 0.22 GB | 31.3× |
| 256 | 32768 | 31.8ms | 11.3ms | 2.8× | 61.9ms | 68.7ms | 0.9× | 27.06 GB | 0.42 GB | **64.5×** |
| 512 | 4096 | 1.0ms | 0.7ms | 1.4× | 1.8ms | 3.0ms | 0.6× | 0.49 GB | 0.12 GB | 4.1× |
| 512 | 8192 | 3.3ms | 1.9ms | 1.7× | 6.3ms | 10.6ms | 0.6× | 1.80 GB | 0.22 GB | 8.2× |
| 512 | 16384 | 12.2ms | 6.4ms | 1.9× | 23.7ms | 41.8ms | 0.6× | 6.93 GB | 0.42 GB | 16.5× |
| 512 | 32768 | 47.8ms | 23.4ms | 2.0× | 93.6ms | 169.9ms | 0.6× | 27.26 GB | 0.82 GB | **33.1×** |

(Small N=256/1024 rows omitted — overhead-bound ~0.5 ms floor, ratios not meaningful.)

### Observations on the gain

- **Memory is the headline win — linear vs quadratic.** Naive peak scales `∝ N²` (each doubling of `N` ≈ 4× memory: 0.46 → 1.75 → 6.83 → 27 GB). Triton scales `∝ N` (≈ 2×: 0.07 → 0.12 → 0.22 → 0.42 GB). At `N=32768` Triton uses **33–65× less memory**, and naive would OOM not much beyond this on a normal GPU while Triton keeps scaling. This — not raw speed — is the point of FlashAttention: it never materializes the `O(N²)` score matrix.
- **Forward is 1.4–2.8× faster**, and the speedup *grows* with `N` (more of the naive cost is `O(N²)` memory traffic that flash avoids). The win is *smaller at larger `Dm`* (2.8× at Dm=256 vs 2.0× at Dm=512) because bigger head dim shifts the work toward compute, where flash's memory-traffic advantage matters less.
- **Backward is *slower* in Triton (0.6–0.9×)** — the flash tradeoff. Flash **recomputes** `S`/`P` in the backward (to avoid storing them), spending extra FLOPs across 3 kernel launches, whereas naive keeps the whole graph materialized and does a recompute-free backward. So flash **trades backward time for memory**: at Dm=512 the backward is ~1.7× slower but uses ~30× less memory. (The kernel is also untuned — no `@triton.autotune` — so some of the backward gap is recoverable.)
- **Net:** flash buys you (1) drastically lower memory → the ability to run long sequences at all, and (2) a faster forward; the backward pays a modest, expected recompute cost for the memory saving. Wall-clock speedup alone understates the value — the memory column is the real story.

### To push further
- **`@triton.autotune`** over `(Q_TILE, K_TILE, num_warps, num_stages)` — biggest lever, especially for the backward.
- Calibrate against **`F.scaled_dot_product_attention`** (production flash) to see the remaining gap.
- Larger `Dm` + longer `N` to probe the compute-bound regime.
