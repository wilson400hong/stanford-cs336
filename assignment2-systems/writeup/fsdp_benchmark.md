# FSDP: Peak Memory vs. World Size

Peak per-GPU memory during training with FSDP on and off, at world size 1, 2 and 4.
Code in `cs336_systems/fsdp.py`, benchmark in `cs336_systems/benchmark/benchmark_fsdp.py`.

## Setup

Model (defaults in `benchmark_fsdp.py`, same as the DDP benchmark):

| | |
|---|---|
| `d_model` | 2560 |
| `d_ff` | 10240 |
| `num_layers` | 32 |
| `num_heads` | 32 |
| `context_length` | 512 |
| `vocab_size` | 10000 |
| `batch_size` | 8 (per rank) |
| dtype | fp32 (no mixed precision) |
| optimizer | AdamW |
| backend | NCCL, 4× GB300 (284 GiB each) |

≈ 3.41B params ≈ 12.7 GiB in fp32.

- **FSDP on**: `FSDPModule`. Linear and Embedding weights are sharded across ranks;
  RMSNorm weights are replicated.
- **FSDP off**: `DDPModule`. Every rank holds full params, grads and optimizer state.
  At world size 1 its all-reduce does nothing useful.

### How peak memory is measured

The model is built on the GPU at full size before `FSDPModule` shards it, so the
benchmark excludes init from the measurement:

1. Build the model, wrap it (FSDP or DDP), create the optimizer.
2. `torch.cuda.empty_cache()` + `torch.cuda.reset_peak_memory_stats()`.
3. Run 2 warmup + 5 timed training steps. The first step allocates the AdamW state, so
   that is included.
4. Report `max_memory_allocated()` / `max_memory_reserved()`, taking the max over ranks.

Columns:
- **Setup**: memory allocated after wrapping, i.e. the params resident on each rank.
- **After step**: memory allocated after `optimizer.step()`: params + grads + AdamW `m`, `v`.
- **Peak alloc**: peak memory held by tensors during training.
- **Peak reserved**: peak memory the caching allocator held from the GPU, including
  fragmentation.

## Results

| FSDP | World size | Setup | After step | Peak alloc | Peak reserved | Step time | Grad-sync wait |
|---|---|---|---|---|---|---|---|
| off | 1 | 12.82 GiB | 50.91 GiB | 92.50 GiB | 112.28 GiB | 1.613 s | 2.7 ms |
| off | 2 | 12.82 GiB | 50.91 GiB | 92.50 GiB | 116.28 GiB | 1.621 s | 2.7 ms |
| off | 4 | 12.82 GiB | 50.91 GiB | 92.50 GiB | 116.28 GiB | 1.624 s | 2.8 ms |
| on  | 1 | 12.82 GiB | 50.91 GiB | 92.88 GiB | 107.68 GiB | 1.633 s | 4.6 ms |
| on  | 2 |  6.41 GiB | 25.46 GiB | 73.57 GiB |  79.37 GiB | 1.605 s | 5.0 ms |
| on  | 4 |  3.30 GiB | 12.84 GiB | **64.06 GiB** | **68.85 GiB** | 1.642 s | 5.0 ms |

Peak allocated, relative to FSDP off:

| World size | FSDP off | FSDP on | Saved |
|---|---|---|---|
| 1 | 92.50 GiB | 92.88 GiB | −0.38 GiB (−0.4%) |
| 2 | 92.50 GiB | 73.57 GiB | 18.93 GiB (20.5%) |
| 4 | 92.50 GiB | 64.06 GiB | 28.44 GiB (30.7%) |

## Analysis

### Model state shrinks as 1/N

With FSDP, **Setup** and **After step** scale as exactly 1/N: 12.82 → 6.41 → 3.30 GiB
and 50.91 → 25.46 → 12.84 GiB. Each rank stores only its slice of the params, grads,
and both AdamW moments, which is 16 bytes/param ÷ N. Without FSDP these numbers stay
the same at every world size, because DDP replicates everything.

### What FSDP shards, and what it doesn't

| Memory | Size (N=1) | Sharded by FSDP? |
|---|---|---|
| Params | 12.8 GiB | yes |
| Grads | 12.8 GiB | yes (reduce-scattered into a shard) |
| AdamW `m`, `v` | 25.6 GiB | yes: the optimizer is built on the sharded params, so its state is shard-sized too |
| Activations | ≈ 54 GiB | **no**: each rank still runs its full batch through every layer |

These runs use **no activation checkpointing**. The current `model.py` has no
checkpointing option, so every layer keeps its forward activations until backward.

### Peak memory doesn't shrink as 1/N: activations dominate

Peak memory is reached at the boundary between forward and backward. At that point
`zero_grad(set_to_none=True)` has freed last step's grads, no new grads exist yet, and
all forward activations are still alive. **Grads are not part of the peak**, so:

```
peak ≈ (params + AdamW m + v) / N  +  activations
     ≈ 3 × setup                   +  activations
     ≈ 38.45 GiB / N               +  ~54 GiB (not sharded)
```

Solving for activations from the measured numbers:

| Config | 3 × setup | Peak alloc | Activations (peak − 3 × setup) |
|---|---|---|---|
| FSDP off, any N | 38.45 GiB | 92.50 GiB | 54.05 GiB |
| FSDP on, N=1    | 38.45 GiB | 92.88 GiB | 54.43 GiB |
| FSDP on, N=2    | 19.22 GiB | 73.57 GiB | 54.35 GiB |
| FSDP on, N=4    |  9.89 GiB | 64.06 GiB | 54.17 GiB |

Activations are a constant **≈ 54 GiB** (≈ 1.7 GiB per layer) in every config.
FSDP shards parameters, not the batch, and each rank still runs a full batch of 8
sequences through every layer. Much of that is the O(T²) attention matrices
(8 × 32 × 512 × 512 × 4 B = 256 MiB per saved copy per layer).

So FSDP only reduces the ~38 GiB of model state. At N=4 it removes 28.8 GiB of that,
and the ~54 GiB of activations stays the same. As N grows, the peak approaches the
activation floor: N=8 would give ≈ 54 + 4.8 ≈ 59 GiB.

### Why N=2 saves ~20%, not 50%

Only 38.45 of the 92.5 GiB peak (≈ 42%) can be sharded at all. Going from N=1 to N=N
saves `38.45 × (1 − 1/N)`:

| N | Sharded part | Activations | Predicted peak | Measured peak | Saved vs. no FSDP (measured) |
|---|---|---|---|---|---|
| 1 | 38.45 GiB | ≈ 54 GiB | 92.5 GiB | 92.5 GiB | — |
| 2 | 19.22 GiB | ≈ 54 GiB | 73.3 GiB | 73.6 GiB | 18.9 GiB (≈ 20%) |
| 4 |  9.61 GiB | ≈ 54 GiB | 63.7 GiB | 64.1 GiB | 28.4 GiB (≈ 31%) |
| ∞ |  0 GiB    | ≈ 54 GiB | 54 GiB   | —        | 38.5 GiB (≈ 42%, upper bound) |

Doubling N halves only the sharded part. Even infinitely many ranks would save at most
≈ 42% here, because activations make up the rest of the peak.

**With activation checkpointing**, activations would drop from ≈ 54 GiB to roughly one
saved layer input per layer, plus one layer's full activations recomputed during
backward. Model state would then be most of the peak, and FSDP's savings would come
much closer to 1/N. That costs one extra forward per layer.

### FSDP's own overhead is small

The ≈ 0.1–0.4 GiB gap between FSDP configs and the 54.05 GiB DDP baseline comes from
FSDP's transient buffers:
- the all-gathered unsharded weights for the current module plus one prefetched module
  (≤ 100 MiB each; the largest are the FFN matrices and `lm_head`/embedding),
- the flat, padded gradient copy each param uses as reduce-scatter input.

At N=1 these buffers are pure overhead, since there's nothing to shard. That's why
FSDP at N=1 is 0.38 GiB above DDP.

### Peak reserved: less fragmentation

DDP reserves 112–116 GiB for 92.5 GiB of live tensors, about 20–24 GiB of cache slack.
FSDP at N=4 reserves 68.85 GiB for 64.06 GiB, under 5 GiB of slack. FSDP's large
per-module buffers are allocated once and resized in place with
`untyped_storage().resize_()`, and the sharded model state is small. That leaves the
caching allocator less to fragment.

### Step time

Step time is about 1.6 s in every config. All-gather and reduce-scatter overlap with
compute through prefetch and async grad hooks. Only 5 ms of reduce-scatter remains
exposed after backward, compared with about 3 ms for DDP's all-reduce. At this model
size, FSDP's extra communication is close to free on NVLink.

## Takeaways

- FSDP divides per-rank **model state** by N exactly: 51 GiB → 12.8 GiB at N=4.
- **Peak** memory falls only 31% at N=4 (92.5 → 64.1 GiB), because the ≈ 54 GiB of
  activations is unaffected by parameter sharding.
- To go lower, combine FSDP with something that reduces activations: activation
  checkpointing, bf16 compute (`--compute_dtype bf16` measured 44.2 GiB peak at N=4),
  or a smaller per-rank batch.
- FSDP at N=1 has no benefit and adds a little memory.

## Reproduce

```bash
for ws in 1 2 4; do
  for f in --fsdp --no-fsdp; do
    uv run python -m cs336_systems.benchmark.benchmark_fsdp $f --world_size $ws \
      --results_file fsdp_results.jsonl
  done
done
```

Add `--mem_snapshot` to dump a rank-0 memory snapshot of the warmup steps to
`--mem_prof_dir`. Open it at https://pytorch.org/memory_viz.
