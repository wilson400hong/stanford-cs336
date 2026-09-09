# Fully-Sharded Data Parallel (FSDP)

Notes on the FSDP implementation for the assignment `fsdp` problem.

## Goal

Turn the data-parallel axis into a *fully-sharded* one: each rank stores only its
slice of every `Linear`/`Embedding` weight (plus the matching slice of the gradient
and optimizer state), and all-gathers the full weight just in time for forward and
backward. Norms and other tiny tensors stay replicated — the fixed latency of a
collective isn't worth it for a few thousand elements.

## Two implementations

| File | Correct? | Memory behavior |
|------|----------|-----------------|
| `fsdp.py`  | yes | Sharded at rest, but the **forward-gathered weight stays resident through backward** (see "autograd saved-tensor problem"). The `free()` in `fwd_post` is intentionally disabled. |
| `fsdp2.py` | yes | **Real memory saving.** Frees the gathered weight after forward and re-materializes it in backward via in-place storage `resize_`. This is the file the adapter imports. |

`fsdp2.py` is the memory-optimized variant; `fsdp.py` is kept as the simpler,
known-good reference.

## Architecture (shared by both)

`FSDPModule(module, compute_dtype=None)` wraps the model. Construction:

1. **`sync_module`** — broadcast all params/buffers from rank 0 so every rank starts
   identical. Must broadcast `param.data` (not the leaf `Parameter`, which would trip
   autograd's leaf-in-place check).
2. **`shard_params`** — for each `Linear`/`Embedding` weight: `flatten` → pad to a
   multiple of `world_size` → contiguous chunk in rank order → each rank keeps its
   shard (`.clone()`, so it owns storage and the full tensor is freed). Record a
   `ParamState` (`shape`, `numel`, `pad_size`, `shard_size`). Replicated params
   (norms) get `ParamState(is_shardable=False)`.
3. **`attach_fsdp_hooks`** — install the hooks below.

### Hooks on sharded modules

| Hook | Job |
|------|-----|
| `forward_pre_hook` | Wait on a prefetched all-gather (or gather synchronously); swap the full weight into `param.data`; save the shard for restore. |
| `forward_hook` (post) | Restore the shard into `param.data`; free the gathered weight; record forward order (first iter); launch async prefetch for the module two ahead. |
| `full_backward_pre_hook` | Re-gather the full weight so autograd has it for this module's backward. |
| `full_backward_hook` (post) | Record backward order; launch backward prefetch. *(May not fire for `Embedding` — see gotchas.)* |
| `post_accumulate_grad_hook` (on weight) | Reduce-scatter the full grad into a shard-shaped grad; restore the shard into `.data`; free the gathered weight; stash the async handle. |

### Replicated (norm) params

A single `post_accumulate_grad_hook` that all-reduces the gradient
(`div_(world_size)` then `SUM` = mean).

### Public interface

- `forward(*inputs, **kwargs)` — prefetch the first two modules, then call the wrapped module.
- `finish_gradient_synchronization()` — wait on every outstanding grad collective
  (reduce-scatters and all-reduces), then free the buffers kept alive for them.
- `gather_full_params()` — all-gather every param back to full shape (test helper);
  keyed by `self.module.named_parameters()` names so they match the non-parallel baseline.

## Key design decisions & gotchas

- **Gradient averaging.** The non-parallel baseline uses a `mean` loss over the full
  batch; each rank sees `1/world_size` of it. So `div_(world_size)` + `SUM` collective
  reproduces the baseline gradient exactly. Same rule for reduce-scatter and all-reduce.
- **Padding symmetry.** The gradient must be flattened+padded with the *same*
  `pad_size` as the weight, so `input.numel() == world_size * shard_size` for
  reduce-scatter. Pad with zeros on the tail (they land in the unused shard region).
- **`.data` swapping** bypasses autograd version tracking and the leaf-in-place
  restriction, which is why gather/restore can mutate the parameter mid-graph.
- **Collectives are in-place on their output buffer**, which must be preallocated,
  contiguous, and correctly sized (`all_gather_into_tensor`: `output = world_size ×
  shard`; `reduce_scatter_tensor`: `input = world_size × output`).
- **Async lifetime.** An async collective's buffers (reduce-scatter input, gather
  output) must stay alive until `.wait()`. They're stashed on `ParamState` and freed
  in `finish_gradient_synchronization`.
- **gloo backend** (used by the tests) — verify collective support; `ReduceOp.AVG`
  and the `*_into_tensor` variants have historically had gaps. Fallbacks:
  all-reduce+slice, and manual `div_`.

### The autograd saved-tensor problem (the crux of `fsdp2`)

When forward runs `einsum(x, weight)`, autograd saves a reference to the **exact
gathered weight tensor's storage** for backward. Naively `free()`-ing that storage
(`resize_(0)`) after forward makes backward read a 0-sized storage and crash. Pointing
`param.data` at a *new* re-gathered buffer in `bwd_pre` does **not** help — autograd
uses the storage it saved, not the current `.data`.

- **`fsdp.py`** sidesteps this by **not** freeing the forward weight; autograd keeps it
  alive until that module's backward completes. Correct, but the (especially early-layer)
  weights stay resident across the whole forward→backward span.
- **`fsdp2.py`** fixes it the way real FSDP does: keep the **same tensor object**
  (`full_tensor_view`, a view sharing the gathered storage) and toggle its underlying
  storage — `resize_(0)` to free after forward, `resize_(padded_bytes)` **in place** +
  gather into that same storage in `bwd_pre`. Because the saved tensor shares that
  storage, reviving it in place makes backward see valid data. This yields the actual
  memory saving.

- **`Embedding` and `full_backward_hook`.** The embedding forward is an integer index
  (`weight[token_ids]`), so no gradient flows to its inputs and `full_backward_hook`
  may not fire — meaning its backward prefetch/recording can be skipped. The sync
  fallback in `bwd_pre` covers correctness. `full_backward_pre_hook` (which keys off
  the output gradient) does fire, so re-gather still happens.

## Test status

- `test_fsdp_correctness[fp32]` — passing on the base implementation.
- `fsdp2.py` (memory variant) — backward re-materialization implemented and reviewed;
  confirm with the runs below.
- **TODO:** `compute_dtype` (fp16) path, `test_fsdp_gradient_sync`, and the 5× race check.

### How to verify

```bash
# fp32 correctness
uv run pytest tests/test_fsdp.py -k "correctness and fp32" -q -s

# gradient shapes/dtypes + replicated-grad equality
uv run pytest tests/test_fsdp.py -k "gradient_sync and fp32" -q

# race check (async reduce-scatter + in-place storage resize is race-prone)
for i in {1..5}; do echo "=== run $i ==="; uv run pytest tests/test_fsdp.py -k fp32 -x -q || break; done

# full suite (fp16 expected to fail until compute_dtype is done)
uv run pytest tests/test_fsdp.py -q
```

To quantify the `fsdp2` memory win, compare `torch.cuda.max_memory_allocated()` against
`fsdp.py` on a larger model (the toy test model is too small to show much).

## Remaining work

1. **`compute_dtype` (mixed precision).** Cast the shard to `compute_dtype` *before*
   gathering (save bandwidth), run forward/backward low-precision, and in the grad hook
   cast the grad **back to fp32 before reduce-scatter** (the test asserts fp32 grads
   regardless of `compute_dtype`). Master weights and the optimizer step stay fp32.
2. **Re-enable backward prefetch in `fsdp2`.** Currently the async gather in the
   backward-post hook is disabled, so backward gathers synchronously (correct, no
   overlap). Re-enabling must gather into the revived in-place storage and set the handle.
3. **Remove debug `print`s** before final runs (multi-rank stdout is noisy).
