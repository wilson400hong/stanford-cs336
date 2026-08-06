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
