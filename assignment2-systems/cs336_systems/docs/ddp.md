# DDP: Reducing Gradient Communication Time

Three implementations of gradient synchronization for data-parallel training, and the
communication cost of each. Code in `cs336_systems/ddp.py`, benchmark in
`cs336_systems/benchmark/benchmark_ddp.py`.

## Setup

Model (defaults in `benchmark_ddp.py`):

| | |
|---|---|
| `d_model` | 2560 |
| `d_ff` | 10240 |
| `num_layers` | 32 |
| `num_heads` | 32 |
| `context_length` | 512 |
| `vocab_size` | 10000 |
| `batch_size` | 8 (per rank) |
| dtype | fp32 |
| backend | NCCL |

Parameter count:

```
per layer:  4 · 2560²        = 26.2M   (q, k, v, o)
          + 3 · 2560 · 10240 = 78.6M   (SwiGLU w1, w2, w3)
          + 2 · 2560         =  5.1K   (RMSNorm)
                             ≈ 104.9M
× 32 layers                  ≈ 3.36B
+ embedding + lm_head + norm ≈  51.2M
                       total ≈ 3.41B params ≈ 13.6 GB in fp32
```

That 13.6 GB is the all-reduce payload every step. Gradients are spread across
**~291 separate tensors** (32 × 9, plus embedding, final norm, and lm_head).

## Results

Mean gradient-synchronization time per step:

| Variant | Comm time | vs. naive |
|---|---|---|
| Naive — one `all_reduce` per parameter | 0.1462 s | 1.0× |
| Flattened — one `all_reduce` for all gradients | 0.0468 s | 3.1× |
| Overlapped — async `all_reduce` per parameter during backward | 0.0028 s | 52× |

## Why flattening helps (3.1×)

The saving is `0.1462 − 0.0468 = 0.0994 s` spread over ~291 collectives, so roughly
**340 µs of fixed cost per `all_reduce`** — kernel launch, NCCL protocol setup, stream
synchronization. That cost is paid per call regardless of message size.

Gradient tensor sizes are extremely skewed: RMSNorm weights are ~10 KB while FFN
matrices are ~100 MB. The tiny tensors pay nearly the same per-call overhead as the
large ones while transferring almost nothing.

Ring all-reduce compounds this. The algorithm chunks the payload across ranks, so a
small tensor becomes an even smaller per-rank chunk and the transfer never reaches
steady-state bandwidth — it is latency-bound, not bandwidth-bound.

Effective bandwidth confirms this. With a ring moving `2(N−1)/N ×` payload:

| Variant | Effective bandwidth (N=2) | (N=4) |
|---|---|---|
| Naive | ~93 GB/s | ~140 GB/s |
| Flattened | ~291 GB/s | ~437 GB/s |

Flattening turns ~291 latency-bound messages into one bandwidth-bound message.

## Why overlapping helps (52×) — and what the number actually means

The overlapped implementation registers a `register_post_accumulate_grad_hook` on every
parameter. As each gradient finishes accumulating during the backward pass, its
`all_reduce` is launched with `async_op=True`; `finish_gradient_synchronization()` then
waits on the collected handles after backward completes.

**The 0.0028 s is not a measurement of communication cost.** Two reasons:

1. **By design**, most of the communication now happens *during* backward, concurrently
   with compute. What remains after backward is only the tail — the last few gradients
   that had no compute left to hide behind.
2. **By artifact**, with NCCL `handle.wait()` synchronizes the CUDA *stream*, not the CPU
   thread. A CPU-side timer around `finish_gradient_synchronization()` can return before
   the collectives have drained.

The arithmetic makes the second point unmistakable: 13.6 GB / 0.0028 s implies
**~4.9 TB/s**, which exceeds any NVLink generation by an order of magnitude. No
interconnect moved that data that fast, so the timer is not measuring the transfer.

**Consequence: comm time is the wrong metric for the overlapped variant.** The correct
comparison is **total step time**, since the entire point is that communication is hidden
inside computation rather than eliminated. Report step time across all three variants.

## Trade-offs

**Flattening costs memory.** The flat buffer is a full extra copy of all gradients
(~13.6 GB here), plus two full copies of traffic — flatten in, unflatten out. Compare
`torch.cuda.max_memory_allocated()` across variants.

**Flattening precludes overlap.** The flat buffer cannot be built until the *last*
gradient exists, so every byte of communication happens strictly after all computation.
Full flattening optimizes the collective while guaranteeing zero overlap. The overlapped
variant makes the opposite trade: ~291 small, inefficient collectives, but they cost
almost nothing in wall-clock because they hide behind backward.

**Bucketing is the middle ground** (not yet implemented). Flatten gradients into
fixed-size buckets so bucket 0 can be communicated while later layers are still in
backward — most of the bandwidth win of flattening, plus overlap. Bucket size is the
tuning knob: larger buckets get better bandwidth, smaller buckets start communicating
earlier.

## Expected scaling

- **World size.** Ring all-reduce moves `2(N−1)/N ×` payload, so going 2 → 6 ranks nearly
  doubles the bytes moved. Communication fraction rises.
- **Model size.** Communication scales with parameter count; compute scales with
  parameters × tokens. Growing `d_model`/`num_layers` at fixed batch size raises the
  communication fraction.
- **Batch size.** Larger per-rank batches amortize the same communication over more
  compute, *lowering* the fraction. This is why production training uses large per-device
  batches.

## Optimizer state sharding (ZeRO-1)

`cs336_systems/sharded_optimizer.py` partitions optimizer state across ranks: each rank
builds an inner optimizer over a disjoint subset of parameter tensors, steps only that
subset, then a single padded `all_gather` restores every rank's full parameter copy.
Parameters and gradients stay replicated; only Adam's `m`/`v` are sharded.

Measured from `_dump_snapshot` on rank 0, `world_size=4`, same model as above
(Ψ ≈ 12.7 GiB of fp32 parameters):

| | vanilla AdamW | ZeRO-1 | delta |
|---|---|---|---|
| **Persistent allocated** (end of run) | 50.91 GiB | 31.97 GiB | **−18.94 GiB (−37%)** |
| Reserved (allocator pool) | 69.35 GiB | 98.33 GiB | +28.98 GiB |
| Overall peak allocated | 66.98 GiB | 66.98 GiB | 0 |

**The steady-state saving matches theory exactly.** Predicted persistent footprint is
`4Ψ` for vanilla (params + grads + `m` + `v`) versus `2Ψ + 2Ψ/N` for ZeRO-1:

```
vanilla:  4 × 12.7                 = 50.8 GiB   (measured 50.91)
ZeRO-1:   2 × 12.7 + 2 × 12.7 / 4  = 31.9 GiB   (measured 31.97)
saving:   2Ψ(1 − 1/N) = 1.5Ψ       = 19.1 GiB   (measured 18.94)
```

Two independent readings agree — summing `active_allocated` blocks in the snapshot's
`segments`, and replaying the `device_traces` allocation events to the end of the run.

### But peak memory did not improve

This is the part worth understanding, and it has two separate causes.

**1. The overall peak is set before any optimizer state exists.** Both runs peak at
66.98 GiB at the *same* trace event, during the first backward pass. Adam allocates `m`
and `v` lazily on the first `step()`, so at that moment neither run has optimizer state —
the peak is parameters + gradients + activations + DDP's flat all-reduce buffer, which
ZeRO-1 does not touch. Nothing about sharding can lower it.

**2. The sharded `step()` allocates a large per-step transient.** The ZeRO-1 trace shows
allocation sizes absent from the vanilla trace:

```
1 × 12.88 GiB   ← DDP's flat gradient buffer (present in both)
5 ×  3.22 GiB   ← ShardedOptimizer: local_flat (1) + gathered (4)
```

That's **16.1 GiB of transient allocation per step**, every step, to reconstitute full
parameters via `all_gather`. It claws back most of the 18.9 GiB of steady-state saving at
the moment of peak usage — and the repeated allocate/free churn is why *reserved* memory
grew by 29 GiB: the caching allocator fragments and never returns blocks to the driver.

**Takeaway:** ZeRO-1 as implemented reduces the memory a model *holds*, not the memory it
*touches*. For fitting a larger model that distinction matters — peak is what OOMs you.
Production ZeRO avoids this by reusing a preallocated flat buffer rather than allocating
per step, and by bucketing the all-gather so only a fraction of parameters is
materialized at once.

### Communication cost

ZeRO-1 adds an all-gather of all parameters (~12.7 GiB payload) every step, on top of the
existing gradient all-reduce. This is the trade: linear reduction in optimizer-state
memory for roughly double the per-step communication.

### Caveats on these numbers

- Peak figures come from replaying `device_traces`. The endpoint values match theory to
  within 0.5%, but mid-trace running totals drift (allocation and free events do not pair
  1:1 in the trace), so **per-step peaks in this table should be confirmed with
  `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()`** before being quoted.
- Rank 0 only. Because the partition is greedy-by-`numel` over whole tensors, other ranks
  may hold slightly different amounts; the largest rank is what determines whether the
  job fits.

## Open items

These affect how defensible the numbers above are:

- **`world_size` for these runs is unconfirmed** (the benchmark default has since changed
  to 4). The bandwidth table gives both N=2 and N=4 for that reason. Flags should be
  recorded alongside each result.
- **No cross-rank aggregation.** Only rank 0's mean is reported. A step is not complete
  until the *slowest* rank finishes, so the reported figure should be the max across
  ranks (an `all_reduce` with `ReduceOp.MAX` on the timing tensor).
- **No results returned to the parent process.** `mp.spawn` returns `None`, so results
  exist only as stdout text and cannot be assembled into a sweep table.
- **Single run, mean only.** No median, stdev, or repeat runs. On a shared GPU host,
  run-to-run variance has already been observed to be large (one step-time reading moved
  0.95 s → 1.64 s with communication time unchanged, pointing at compute-side contention
  or allocator pressure rather than a communication change).
- **Overlapped step time not yet measured** — the metric that actually matters for that
  variant.
- **Per-step peak memory** for the ZeRO-1 comparison needs `max_memory_allocated()`
  rather than trace replay (see caveat above).
- **ZeRO-1 step time not yet measured.** The extra all-gather should show up as a slower
  step; the memory saving is only interesting alongside its time cost.
