# Problem (scaling_laws): Strategy for constructing the scaling law

Goal: pick the config (architecture + optimizer + batch size + `total_train_tokens`) that gets the lowest
validation loss in **48 B200-hours** (172,800 s), and predict that loss, using at most **12 B200-hours**
(43,200 s = 720 min) of probe runs.

> All absolute numbers below (FLOP/s, minutes per run, model sizes) are **illustrative**, based on an assumed
> throughput table (§2). After Phase 1 measures real throughput, rescale every level/run with the same formulas.

---

## 0. Facts from the API / training code that drive the plan

| Fact (source) | Consequence |
|---|---|
| Budget is **wall-clock seconds**, not FLOPs | Compute = time × throughput, and throughput depends on shape and batch size. Measure it. The real objective is *loss per GPU-second*, not loss per FLOP. |
| Charged time = training loop **including the first-chunk JIT compile** (`training/run.py:348`) | ~1 min fixed overhead per run. Don't run anything shorter than ~3 min. |
| Timeout ⇒ charged full `max_runtime_seconds`, no final loss. No cancel endpoint. | Reserve `max_runtime_seconds ≈ 1.4 × predicted`. Unused time is refunded, so over-reserving costs nothing (but it does block submitting other runs while queued). |
| Queued/running runs reserve their full `max_runtime_seconds` | Submit in waves sized to the remaining budget. |
| You choose `total_train_tokens` for the final run | If it doesn't fit in 48 h it fails. You need an accurate tokens/s for the **exact** final config. |
| SwiGLU MLP (`gate/up/down_proj`); GQA rejected; head_dim must be even | Use `intermediate_size ≈ 8/3·d` ⇒ ~12·L·d² non-embedding params. `num_key_value_heads = num_attention_heads`. |
| Single LR for all params; no per-layer multipliers | μP is not available ⇒ **fit LR as a function of size**. |
| `nn.Linear(..., dtype=config.jax_dtype)` ⇒ with `"bfloat16"` the **parameters are stored in bf16** | Late in a long cosine run, tiny updates can round away. Test bf16 vs fp32 once (loss *and* speed). |
| `optax.adamw` with no mask ⇒ weight decay applies to embeddings and norms too | Keep WD small-ish; test 0.01 vs 0.1 once. |
| Data order is fixed and deterministic; `model_seed` changes only init | Seed reruns measure init noise only. Every config sees the same tokens. |
| Validation = fixed 2^18 tokens (512 sequences); identical config ⇒ HTTP 409 | All runs are compared on the same val set (low *relative* noise). To replicate, change `model_seed`. |
| `total_train_tokens % (512·bs) == 0` and `total_steps % n_evals == 0` | Round tokens to a multiple of `512 · bs · n_evals` (helper in §8). |
| Budget ratio: 12 h probes vs 48 h target = 25% | Very generous. The largest probe can be ~1/40 of the final compute, so extrapolation is less than 2 decades. **The main risks are hyperparameters at scale and throughput misprediction, not the power-law extrapolation.** |

---

## 1. Core principles

1. **Every hyperparameter is a deterministic rule of (N, C).** The *same* rules generate the probes and
   the final run. The scaling law then describes the recipe you will actually use, and nothing is hand-tuned
   only at the end.
   - `shape(N)`: d/L ratio, head_dim, d_ff (§5)
   - `lr(N)`: fitted power law (Phase 2)
   - `bs(C)`: batch-size rule (Phase 2 + prior)
   - fixed: warmup 0.05, final_lr_frac 0.1, betas (0.9, 0.95), clip 1.0, WD from Phase 2
2. **Fit on completed runs only.** Under a cosine schedule an intermediate checkpoint ≠ a shorter run
   (Chinchilla's key point). Use `val_losses` only to spot divergence or instability.
3. **Work in waves, refit between waves.** Never queue the whole plan up front. Re-center each isoFLOP
   level on the latest N_opt prediction.
4. **Write predictions down before you see results** (especially for the holdout run). That is the only
   honest estimate of your extrapolation error.
5. **Keep one run log** (§9) from the first run. The writeup must be reproducible from it.

---

## 2. Compute accounting

Per-token training FLOPs (forward + backward = 3 × forward), with V = 32,000 and seq = 512:

```
flops_per_token = 6·L·(4·d² + 3·d·d_ff)   # attention projections + SwiGLU MLP
                + 6·d·V                    # LM head (untied output projection)
                + 12·L·seq·d               # attention scores QKᵀ and AV
C = flops_per_token × total_train_tokens
```

Why not just `6·(12Ld²)·D`: at small scale the LM head dominates. For d = 512, L = 8 the head is ~98 MFLOP/token
against ~151 MFLOP/token for the blocks. Using 12Ld² alone would bias the fitted exponents.

- Report **N = 12·L·d²** (non-embedding params), as the assignment asks.
- Use the accurate **C** above for every fit.
- Robustness check: also fit with N_total (including embeddings and head), and keep whichever gives the lower
  backtest error (§7.4).

**Illustrative throughput assumption** (replace after Phase 1):

| Non-embedding N | Effective FLOP/s (bf16) |
|---|---|
| ≤ 30M | 2.5e14 |
| 30M–150M | 4e14 |
| 150M–500M | 6e14 |
| ≥ 1B | 9e14 (≈40% of ~2.25 PFLOP/s dense bf16) |

Under this assumption **C_final ≈ 172,800 s × 9e14 ≈ 1.5e20 FLOPs**. The Chinchilla prior (D ≈ 20N) then
gives N_opt ≈ 1.1B and D ≈ 22B tokens. This is only the prior. The probes decide.

---

## 3. Budget allocation (720 min)

| Phase | Purpose | Minutes | Share |
|---|---|---|---|
| 1 | Throughput, compile overhead, memory | 20 | 3% |
| 2 | Hyperparameter rules at small scale | 120 | 17% |
| 3 | IsoFLOP sweep: 405 planned + 65 adaptive fill-ins | 470 | 65% |
| 4a | Holdout run (tests the extrapolation) | 80 | 11% |
| 4b | Throughput of final candidate configs | 10 | 1% |
| — | Reserve (crashes, timeouts, reruns) | 20 | 3% |
| **Total** | | **720** | |

---

## 4. Phases in detail

### Phase 1: throughput, overhead and memory (20 min, 1 wave, all in parallel)

About 90 s of training each, `max_runtime_seconds` ≈ 240. Choose `total_train_tokens` from the guessed
throughput.

| Run | d | L | heads × head_dim | bs (seqs) | N (12Ld²) | Purpose |
|---|---|---|---|---|---|---|
| T1 | 384 | 6 | 6×64 | 128 | 10.6M | small-model speed |
| T2 | 384 | 6 | 6×64 | 128 | 10.6M | T1 at 2× tokens ⇒ fixed overhead = 2·t(T1) − t(T2) |
| T3 | 768 | 12 | 12×64 | 256 | 85M | |
| T4 | 1024 | 16 | 8×128 | 256 | 201M | |
| T5 | 1536 | 20 | 12×128 | 256 | 566M | |
| T6 | 2048 | 20 | 16×128 | 256 | 1.0B | final-scale speed + memory check |
| T7 | 2048 | 20 | 16×128 | 512 | 1.0B | batch-size effect on speed, memory ceiling |

Outputs:
- `overhead_s`, the compile/startup constant.
- `tokens_per_s(shape, bs)` and `flops_per_s = tokens_per_s × flops_per_token`. Fit
  `log(flops_per_s)` against `log N` with a straight line or a lookup table.
- Which configs OOM. A failure charges only the time actually used.
- A first estimate of `C_final`. Recompute it once the final size is known.

### Phase 2: hyperparameter rules (120 min, 2 waves)

Two base sizes, both trained at D = 20N with bs = 128 and `n_evals` = 16:
- **A**: d = 384, L = 6, 6×64 heads, d_ff = 1024, N = 10.6M, D ≈ 212M tokens, ≈ 3 min/run
- **B**: d = 512, L = 8, 8×64 heads, d_ff = 1408, N = 25M, D ≈ 504M tokens, ≈ 10 min/run

**Wave 2a (all at size A, ≈ 45 min)**

| Experiment | Runs | ~min | Detail |
|---|---|---|---|
| LR sweep A | 5 | 16 | peak LR ∈ {1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2} |
| Aspect ratio | 2 | 7 | Same N, d/L ≈ 32 and ≈ 128 (best LR from the sweep, or the 4e-3 guess) |
| Weight decay | 1 | 3 | WD = 0.1 vs 0.01 at the best LR |
| dtype | 1 | 8 | `float32` vs `bfloat16`. fp32 is slower; record both loss and tokens/s |
| Seed noise | 2 | 7 | Best-LR config with `model_seed` = 1, 2 ⇒ σ_seed |
| (optional) final_lr_frac | 1 | 3 | 0.0 vs 0.1 |

**Wave 2b (size B, ≈ 77 min)**, centered on A's results:

| Experiment | Runs | ~min | Detail |
|---|---|---|---|
| LR sweep B | 5 | 51 | 5 LRs spaced 2×, centered at lr_A·(N_B/N_A)^(-0.3) |
| Batch size | 2 | 16 | bs = 256 and 512 at best LR × √(bs/128). Same tokens, so fewer steps |
| Seed noise | 1 | 10 | Best-LR config, `model_seed` = 1 |

**Analysis**
- **Best LR per size.** Fit a parabola of loss against log(LR) through the 5 points:
  `lr* = exp(−b / 2a)`. If the minimum sits at the edge of the sweep, extend the sweep before trusting it.
- **LR rule.** `lr(N) = lr_A · (N / N_A)^(−α)`. Two sizes give a weak α. Phase 3 adds two more points (§4,
  LR checks). Sanity prior: exponents are usually modest. DeepSeek LLM reports η_opt ∝ C^(−0.125), roughly
  N^(−0.25). Be skeptical if the 2-point α is outside about 0.1–0.6.
- **Batch rule `bs(C)`.** Use the largest batch that doesn't cost more than ~σ_seed in loss at equal tokens.
  The useful batch size grows with compute (DeepSeek LLM: B_opt ∝ C^0.33 in tokens). Use a power-of-2 rule,
  for example: `bs = 128` for C < 3e17, `256` up to 1e18, `512` above that, and larger for the final run if
  Phase 1 shows a speed gain and memory allows. Bigger batches also raise FLOP/s, which matters for a
  wall-clock budget.
- **Shape rule.** Pick the d/L value inside the flat region that gives the highest tokens/s.
- **Noise.** σ_seed sets how many points per isoFLOP curve you need and the Huber δ in §7.2. Expect roughly
  0.005–0.02.

### Phase 3: IsoFLOP sweep (470 min, 4–5 waves)

**Levels** spaced ~2.5×. 5 sizes per level at **{0.33, 0.6, 1.0, 1.7, 3.0} × N_center**, so log-uniform with
steps of about 1.75×. Each size gets `D = C / flops_per_token(shape)`, `lr = lr(N)` and `bs = bs(C)`.

| Level | C (FLOPs) | Chinchilla N_center (prior) | D at center | ~min/run | Runs | ~min |
|---|---|---|---|---|---|---|
| L1 | 6.0e16 | 22M | 0.45B | 5 | 5 | 25 |
| L2 | 1.5e17 | 35M | 0.71B | 7 | 5 + 2 LR checks | 50 |
| L3 | 3.75e17 | 56M | 1.1B | 17 | 5 + 2 LR checks | 117 |
| L4 | 1.0e18 | 91M | 1.8B | 43 | 5 | 213 |
| Adaptive | — | — | — | — | ~3–4 | 65 |
| **Total** | | | | | | **470** |

(N_center from `C = 6·N·20N`. After L1, use **your own** fitted N_opt(C) instead.)

**Order**
1. **Wave 3a: L1.** Fit a parabola, get N*(L1).
2. **Wave 3b: L2 + LR checks.** Center L2 by extrapolating from the prior and N*(L1). The LR checks run the
   center model at 0.5× and 2× of lr(N), giving a third point for the LR rule.
3. **Wave 3c: L3 + LR checks.** Refit N_opt(C) and lr(N) with 4 sizes (A, B, L2 center, L3 center).
4. **Wave 3d: L4.** Use the refined rules.
5. **Wave 3e: adaptive fill-ins** (see the decision rules in §6).

**Per-run hygiene**
- `max_runtime_seconds = 1.4 × (overhead_s + C / flops_per_s(shape, bs))`, capped by the remaining budget.
- `n_evals = 16`. Glance at the curve for spikes or divergence. Rerun a diverged config at 0.5× LR from the
  reserve. Don't drop it silently; note it in the log.

### Phase 4a: holdout validation (80 min)

- Before submitting: refit everything (§7) on Phase 2 + Phase 3 data. Predict **N_opt** and **L_opt** at
  `C_h ≈ 1.8e18` (≈ 1.8× L4). **Write the predicted loss into the log.**
- Run that config: shape(N_opt), lr(N_opt), bs(C_h), D = C_h / flops_per_token. About 76 min.
- The residual `r = L_obs − L_pred` is your empirical extrapolation error. Then add the run to the fit.
- If |r| > ~3σ_seed, investigate before going further (the LR rule is the usual cause).

### Phase 4b: final-config throughput (10 min)

- Pick the top 2–3 candidate shapes for the final run (§7.5). Run each for about 2–3 min.
- From each, measure tokens/s (subtracting `overhead_s`) and confirm the memory fits.
- These measured tokens/s set `total_train_tokens` for the final run.

### Reserve (20 min)

Crashes, timeouts, and reruns of diverged runs. If nothing goes wrong, spend it on one more isoFLOP point at
L4 where the parabola is least constrained.

---

## 5. Architecture rule `shape(N)`

- **d/L ratio:** from Phase 2 (Kaplan: flat for d/L ≈ 30–150). Within that band, prefer wider and shallower
  because it runs faster.
- **head_dim:** 64 for d < 1024, 128 for d ≥ 1024. `num_key_value_heads = num_attention_heads`.
- **d_ff:** `round_up(8·d/3, 128)`, or a multiple of 256 at large d, for matmul efficiency.
- **d:** a multiple of 128 (64 for tiny models). Adjust L so that 12·L·d² ≈ target N.
- Fixed: `rope_theta` = 1e6, `rms_norm_eps` = 1e-6, `attention_bias` = False, `tie_word_embeddings` = False,
  `vocab_size` = 32000, `dtype` per Phase 2.

Example final candidates if N_opt ≈ 1B (illustrative):

| d | L | heads×hd | d_ff | N = 12Ld² | d/L |
|---|---|---|---|---|---|
| 1792 | 24 | 14×128 | 4864 | 925M | 75 |
| 2048 | 20 | 16×128 | 5632 | 1.01B | 102 |
| 2304 | 18 | 18×128 | 6144 | 1.15B | 128 |

---

## 6. Decision rules during the sweep

| Situation | Action |
|---|---|
| LR-sweep minimum at the edge | Add 1–2 LRs past that edge before trusting lr* |
| IsoFLOP minimum at the smallest or largest size in a level | Add one size 1.75× further on that side (adaptive budget) |
| IsoFLOP parabola has a ≤ 0 (no curvature) or a poor fit | Points too close together or too noisy: add a size at 0.2× or 5×, or a seed rerun |
| σ_seed > 0.02 | Fit pooled across levels (parametric fit, §7.2) rather than per-level minima |
| A run diverges or spikes | Rerun at 0.5× LR from the reserve. If it recurs at large N, lower the LR prefactor for all later runs |
| Measured runtime > 1.2× predicted | Refit the throughput model before the next wave |
| Holdout residual > 3σ_seed | Check the LR rule at holdout scale first; consider ±LR reruns at L4 instead of fill-ins |

---

## 7. Fitting methods (use several, compare, ensemble)

### 7.1 IsoFLOP minima (Hoffmann approach 2)

For each level, fit `L = a·(log N)² + b·log N + c`. Then `log N* = −b / (2a)` and `D* = C / flops_per_token(N*)`.
Fit `N* ∝ C^a` and `D* ∝ C^b` in log space (`np.polyfit(log C, log N*, 1)`). Check that a + b ≈ 1.

```python
def isoflop_min(logN, L):
    a, b, c = np.polyfit(logN, L, 2)
    assert a > 0, "no curvature"
    x = -b / (2 * a)
    assert logN.min() <= x <= logN.max(), "minimum outside the grid"
    return np.exp(x), np.polyval([a, b, c], x)   # N*, L*
```

### 7.2 Parametric fit (Hoffmann approach 3)

`L(N, D) = E + A / N^α + B / D^β`, fit on **all** completed Phase 2–4a runs.

```python
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import huber, logsumexp

def fit_chinchilla(N, D, L, delta=1e-3):
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def obj(p):
        a, b, e, alpha, beta = p
        pred = logsumexp(
            [a - alpha * logN, b - beta * logD, np.full_like(logN, e)], axis=0
        )
        return huber(delta, pred - logL).sum()

    best = None
    grid = itertools.product(
        range(0, 30, 5), range(0, 30, 5), [-1, -0.5, 0, 0.5, 1],
        [0, 0.5, 1, 1.5, 2], [0, 0.5, 1, 1.5, 2],
    )
    for p0 in grid:
        r = minimize(obj, p0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    a, b, e, alpha, beta = best.x
    return dict(A=np.exp(a), B=np.exp(b), E=np.exp(e), alpha=alpha, beta=beta)
```

- Set δ to about σ_seed / L (relative noise in log space).
- Minimizing `L(N, C / flops_per_token(N))` over N gives N_opt(C) and L_opt(C) directly. Using the accurate
  FLOP count makes this numerical 1-D minimization rather than the closed form, which assumes C = 6ND.
- Note: with only ~10–150M params and a fixed val set, E and α/β can trade off against each other. Report
  parameter uncertainty from bootstrap resampling of the runs.

### 7.3 Loss vs. compute

Fit `L*(C) = E + k · C^(−γ)` to the isoFLOP minima from §7.1, plus the holdout. Keep E: a pure power law
overpredicts how far the loss keeps falling.

### 7.4 Backtest (how much to trust the extrapolation)

- Refit §7.1–7.3 **without L4** and predict L4's N* and L*. Compare with the observed values.
- Refit **with** L4 and predict the holdout (done for real in Phase 4a).
- The errors grow with extrapolation distance. The final run is ~80× beyond the holdout in compute, so expect
  the final loss error to be somewhat larger than the holdout residual.

### 7.5 Choosing the final config (wall-clock aware)

Because the budget is time, optimize over **candidate shapes** directly, not over N alone:

```python
for shape in candidates:                      # §5 shapes around N_opt, ×0.5 … ×2
    tps = measured_or_modeled_tokens_per_s(shape, bs_final)
    D = round_tokens(0.95 * tps * (172_800 - overhead_s), bs_final, n_evals)
    L_hat = L_param(N=12 * shape.L * shape.d**2, D=D)  # §7.2
pick argmin L_hat; check that N is within ~1.5× of approach 2's N_opt(C_final)
```

A slightly smaller but faster shape often wins under a wall-clock budget, because the loss curve is flat near
N_opt. The 0.95 factor is a safety margin against throughput noise.

### 7.6 Final loss prediction

1. Evaluate both `L_param(N_final, D_final)` (§7.2) and `L*(C_final)` (§7.3) with C_final from the measured
   FLOP/s.
2. Average the two. If they differ by more than ~0.05, prefer the parametric fit, since it uses all runs.
3. Apply a bias correction from the holdout. With the holdout already included in the refit, apply only
   whatever residual bias remains.
4. Note in the writeup that the leaderboard evaluates on more validation tokens than the API's 2^18. Expect
   small noise around the prediction.

---

## 8. Final hyperparameters and submission

| Field | Rule |
|---|---|
| Architecture | Winner of §7.5 |
| `dtype` | bf16 unless Phase 2 showed fp32 is meaningfully better *per GPU-second* |
| Peak LR | `lr(N_final)` from the 4-point fit, nudged ~0.8× for safety (a diverged 48 h run is the worst outcome) |
| `train_batch_size` | `bs(C_final)`, capped by memory from Phase 1 / 4b. Typically 512–2048 sequences at this scale |
| warmup / final_lr_frac / WD / betas / clip | Phase 2 values (defaults 0.05 / 0.1 / 0.01 / (0.9, 0.95) / 1.0 unless changed) |
| `total_train_tokens` | `round_tokens(0.95 × measured tokens/s × (172,800 − overhead), bs, n_evals)` |
| `n_evals` | 16 (eval cost is small, about 2^18 tokens each) |
| `predicted_final_loss` | From §7.6 |

```python
def round_tokens(target, bs, n_evals=16, seq=512):
    unit = seq * bs * n_evals        # satisfies both divisibility constraints
    return max(1, int(target // unit)) * unit
```

**Open question to check in the API docs:** what `max_runtime_seconds` the final submission should carry,
probably 172,800. The probe API caps it at 12 h, but the final run is executed separately.

Submit via `POST /final_submission`. You can resubmit; the latest submission wins.

---

## 9. Run log schema (one row per experiment)

`experiment_id, phase, wave, d, L, n_heads, head_dim, d_ff, bs, peak_lr, wd, dtype, seed, total_train_tokens,
N_nonembed, N_total, flops_per_token, C, predicted_runtime_s, used_runtime_s, tokens_per_s, flops_per_s,
status, final_val_loss, val_losses, predicted_loss (if any), notes`

Keep it as a CSV or JSON next to the fitting scripts. Every plot and table in the writeup is generated from it.

---

## 10. Timeline / checklist

- [ ] Offline (0 budget): config generator, FLOP counter, `round_tokens`, run logger, fitting scripts (test
      them on `data/isoflops_curves.json`)
- [ ] Wave 1: Phase 1 throughput (7 runs) → throughput model, `overhead_s`, memory limits
- [ ] Wave 2a: size-A sweeps → lr_A, d/L, WD, dtype, σ_seed
- [ ] Wave 2b: size-B sweeps → lr_B, α (2-point), batch rule
- [ ] Wave 3a: L1 → N*(L1)
- [ ] Wave 3b: L2 + LR checks → refit
- [ ] Wave 3c: L3 + LR checks → 4-point LR rule, refit N_opt(C)
- [ ] Wave 3d: L4
- [ ] Wave 3e: adaptive fill-ins; backtest (L1–L3 → L4)
- [ ] Write down holdout prediction → Wave 4a: holdout run → residual
- [ ] Wave 4b: throughput of 2–3 final candidates
- [ ] §7.5 shape selection, §7.6 loss prediction, `POST /final_submission`
- [ ] Writeup

---

## 11. Writeup outline (maps to the assignment's questions)

1. **Which runs and why.** Budget table (§3), wave structure, and why the isoFLOP levels were placed where
   they were. Include the fact that the budget is wall-clock and how that changed the design.
2. **How the law was fit.** FLOP accounting (§2); approaches 2 and 3 (§7.1–7.2); loss vs. compute (§7.3);
   how the LR and batch rules were derived.
3. **Fit quality.** IsoFLOP plots with parabolas; N_opt(C) and D_opt(C) plots; parametric-fit residuals;
   backtest and holdout errors; bootstrap intervals on the exponents.
4. **Prediction at 48 h.** C_final from measured FLOP/s, N_opt, D_opt, shape choice (§7.5), predicted loss
   with an uncertainty band.
5. **Final hyperparameters.** The table from §8, with each value traced back to the run(s) that justify it.

---

## 12. Risk register

| Risk | Mitigation |
|---|---|
| LR rule extrapolates badly to ~1B | 4-point fit, LR checks at L2/L3, holdout, 0.8× safety factor |
| Final run exceeds 48 h | Measure exact-config tokens/s (Phase 4b), 0.95 margin, subtract overhead |
| OOM at final batch size | Phase 1 T6/T7 and Phase 4b memory checks; fall back to a smaller batch |
| Too few points or too noisy per isoFLOP level | Parametric fit pools all runs; adaptive fill-ins; σ_seed measured |
| bf16 parameter precision stalls long runs | Phase 2 dtype test; if fp32 is too slow, keep bf16 and keep final_lr_frac ≥ 0.1 |
| Grid not centered on the optimum | Lowest-level-first waves with refit and re-centering |
| Budget overrun from timeouts | 1.4× runtime reservation, refits of the throughput model, 20 min reserve |
