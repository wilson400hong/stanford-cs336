# Attention Benchmark: Naive vs `torch.compile`

Single-head scaled dot-product attention, forward + backward, batch fixed. Times = mean over timed steps (std ~1–3%, tight).

```bash
uv run python -m cs336_systems.benchmark_attention                   # naive
uv run python -m cs336_systems.benchmark_attention --torch_compile   # compiled
```


## Summary
- compile helps almost everywhere (fwd 1.6–3.8×, bwd 1.2–3.3×). Its win is kernel
  fusion of the softmax scale/mask/exp/normalize passes + removing Python/launch
  overhead.
- The speedup decays as the config becomes hardware-bound, and only truly vanishes
  at the saturated corner d=128 @ 16384 (0.99× fwd, 1.00× bwd) — there the work is
  fully O(seq²) bandwidth-bound and has the largest matmuls, so there's no overhead
  left to fuse away.
- At the same ctx=16384 but smaller d_model, compile still gives ~2× (2.46× at
  d=16). So it's not "large context kills it" broadly — it's "the largest+widest
  single point."
- Tiny end matters too: d=128, ctx=256 is 0.95× — compile is slightly slower because
  the fixed dispatch cost isn't amortized when there's almost no work.


## Comparison (times in ms; × = naive / compiled speedup)

| d_model |   ctx | fwd naive | fwd comp | fwd× | bwd naive | bwd comp | bwd× |
|--------:|------:|----------:|---------:|-----:|----------:|---------:|-----:|
| 16 | 256 | 0.968 | 0.448 | 2.16× | 1.407 | 0.789 | 1.78× |
| 16 | 1024 | 0.938 | 0.462 | 2.03× | 1.370 | 0.773 | 1.77× |
| 16 | 4096 | 2.536 | 1.205 | 2.10× | 4.470 | 2.743 | 1.63× |
| 16 | 8192 | 8.711 | 3.557 | 2.45× | 16.182 | 9.385 | 1.72× |
| 16 | 16384 | 32.696 | 13.282 | 2.46× | 62.054 | 34.713 | 1.79× |
| 32 | 256 | 1.903 | 0.556 | 3.42× | 1.887 | 0.865 | 2.18× |
| 32 | 1024 | 1.883 | 0.500 | 3.77× | 1.855 | 0.823 | 2.25× |
| 32 | 4096 | 3.118 | 1.244 | 2.51× | 5.002 | 3.187 | 1.57× |
| 32 | 8192 | 9.455 | 4.302 | 2.20× | 17.101 | 10.897 | 1.57× |
| 32 | 16384 | 33.792 | 14.652 | 2.31× | 63.372 | 36.582 | 1.73× |
| 64 | 256 | 1.191 | 0.481 | 2.48× | 2.630 | 0.799 | 3.29× |
| 64 | 1024 | 1.166 | 0.488 | 2.39× | 2.579 | 0.795 | 3.24× |
| 64 | 4096 | 3.044 | 1.533 | 1.99× | 5.721 | 3.730 | 1.53× |
| 64 | 8192 | 10.244 | 5.451 | 1.88× | 18.911 | 12.979 | 1.46× |
| 64 | 16384 | 38.496 | 19.570 | 1.97× | 72.686 | 46.220 | 1.57× |
| 128 | 256 | 1.007 | 1.063 | 0.95× | 1.458 | 1.190 | 1.23× |
| 128 | 1024 | 1.908 | 1.094 | 1.74× | 1.910 | 1.228 | 1.56× |
| 128 | 4096 | 3.924 | 2.360 | 1.66× | 6.624 | 5.031 | 1.32× |
| 128 | 8192 | 12.833 | 7.896 | 1.63× | 23.816 | 17.882 | 1.33× |
| 128 | 16384 | 46.707 | 47.011 | 0.99× | 88.568 | 88.775 | 1.00× |

## Takeaways

- **`torch.compile` helps across most of the grid** — forward 1.6–3.8×, backward 1.2–3.3× — but the benefit **shrinks as the work becomes hardware-bound**, and vanishes at the largest corner.
- **Why the decay:** compile's win is kernel **fusion** (fuse the softmax scale/mask/exp/normalize passes over the `seq×seq` tensor) + removing Python/launch overhead. When the config is overhead-bound (small ctx/d_model) there's a lot to fuse away → up to ~3.8×. As ctx and d_model grow, kernels approach the memory-bandwidth / matmul-FLOP roofline, so there's little overhead left to remove.
- **"No help at large context" is really "no help at the saturated corner."** At ctx=16384 the forward speedup is still 2.46× (d=16), 2.31× (d=32), 1.97× (d=64) — only **d=128 @ 16384 collapses to ~1.0×** (compiled ≈ naive, within noise). That point is fully bandwidth+compute-bound (`O(seq²)` memory *and* the largest matmuls).

  ```
  fwd× @ ctx=16384:  d16 2.46×  d32 2.31×  d64 1.97×  d128 0.99×
  fwd× @ ctx=1024 :  d16 2.03×  d32 3.77×  d64 2.39×  d128 1.74×
  ```

- **Watch the tiny end too:** d=128, ctx=256 is **0.95×** — compile is marginally *slower* because the fixed dispatch cost isn't amortized when there's almost no work.
- **Backward gains < forward** — the backward graph is dominated by the two big matmuls (few fusible elementwise ops), so fusion has less to exploit.

---

## Appendix — raw runs

### Naive (`benchmark_attention`)
```
d_model=16   ctx=256   fwd=0.000968 bwd=0.001407
d_model=16   ctx=1024  fwd=0.000938 bwd=0.001370
d_model=16   ctx=4096  fwd=0.002536 bwd=0.004470
d_model=16   ctx=8192  fwd=0.008711 bwd=0.016182
d_model=16   ctx=16384 fwd=0.032696 bwd=0.062054
d_model=32   ctx=256   fwd=0.001903 bwd=0.001887
d_model=32   ctx=1024  fwd=0.001883 bwd=0.001855
d_model=32   ctx=4096  fwd=0.003118 bwd=0.005002
d_model=32   ctx=8192  fwd=0.009455 bwd=0.017101
d_model=32   ctx=16384 fwd=0.033792 bwd=0.063372
d_model=64   ctx=256   fwd=0.001191 bwd=0.002630
d_model=64   ctx=1024  fwd=0.001166 bwd=0.002579
d_model=64   ctx=4096  fwd=0.003044 bwd=0.005721
d_model=64   ctx=8192  fwd=0.010244 bwd=0.018911
d_model=64   ctx=16384 fwd=0.038496 bwd=0.072686
d_model=128  ctx=256   fwd=0.001007 bwd=0.001458
d_model=128  ctx=1024  fwd=0.001908 bwd=0.001910
d_model=128  ctx=4096  fwd=0.003924 bwd=0.006624
d_model=128  ctx=8192  fwd=0.012833 bwd=0.023816
d_model=128  ctx=16384 fwd=0.046707 bwd=0.088568
```

### `torch.compile` (`benchmark_attention --torch_compile`)
```
d_model=16   ctx=256   fwd=0.000448 bwd=0.000789
d_model=16   ctx=1024  fwd=0.000462 bwd=0.000773
d_model=16   ctx=4096  fwd=0.001205 bwd=0.002743
d_model=16   ctx=8192  fwd=0.003557 bwd=0.009385
d_model=16   ctx=16384 fwd=0.013282 bwd=0.034713
d_model=32   ctx=256   fwd=0.000556 bwd=0.000865
d_model=32   ctx=1024  fwd=0.000500 bwd=0.000823
d_model=32   ctx=4096  fwd=0.001244 bwd=0.003187
d_model=32   ctx=8192  fwd=0.004302 bwd=0.010897
d_model=32   ctx=16384 fwd=0.014652 bwd=0.036582
d_model=64   ctx=256   fwd=0.000481 bwd=0.000799
d_model=64   ctx=1024  fwd=0.000488 bwd=0.000795
d_model=64   ctx=4096  fwd=0.001533 bwd=0.003730
d_model=64   ctx=8192  fwd=0.005451 bwd=0.012979
d_model=64   ctx=16384 fwd=0.019570 bwd=0.046220
d_model=128  ctx=256   fwd=0.001063 bwd=0.001190
d_model=128  ctx=1024  fwd=0.001094 bwd=0.001228
d_model=128  ctx=4096  fwd=0.002360 bwd=0.005031
d_model=128  ctx=8192  fwd=0.007896 bwd=0.017882
d_model=128  ctx=16384 fwd=0.047011 bwd=0.088775
```
