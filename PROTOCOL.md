# Evaluation protocol — Velo Base 1.0

Benchmark: `LocalLLaMA/typed-decisions`, revision `ea9306458d6e9563628369a3d1e72e362fb381d2` (Apache-2.0). Dataset rows are not shipped; `velo_base.data.load_splits` downloads that revision (or reads a local cache under `data/typed_decisions/`).

## Splits and sealing

| split | cases | decisions | use |
|---|---|---|---|
| official `train` | 1,200 | 6,000 | fitting, model selection, calibration |
| ↳ OOS / validation holdout | hash-bucket subset of train | — | temperature and mode selection only |
| official `test` | 400 | 2,000 | sealed evaluation after freeze |

Train/validation rule used for calibration fitting:

```text
int(sha256(case_id.encode()).hexdigest(), 16) % 100 < 20   → validation
otherwise                                                 → training part
```

No seed and no shuffle for that split. The sealed test split is read for scoring after architecture, hyper-parameters, and calibration are frozen. Record: `artifacts/seal_v1.json` (model id `velo-base-1.0`).

Rules observed for the seal:

- Never train on the official test split.
- Never use test outputs for architecture, hyper-parameter, or calibration choices.
- Calibration (per-qtype temperature) was fitted on an OOS case holdout from train; temperatures preserve argmax.
- Default inference mode is `calibrated` (`clip_renorm` + per-qtype T). Mode `clip` is the Brier-oriented companion (same heads, T = 1). Mode `raw_soft_norm` is legacy and not recommended.

## I/O contract

**Input (per case)**

| field | type | notes |
|---|---|---|
| `workflow` | string | One of the four training workflows, or a vertical id after refit |
| `state` | JSON object (or JSON string) | Business payload; may be nested / wrapped — `prepare_state` normalizes presentation |
| `questions` | object map `qid → {type, criteria, instructions}` | `type` ∈ {`choice`, `noul`, `score`} |

Gold `factors` / labels are **not** read at inference.

**Output (per question id)**

| field | meaning |
|---|---|
| `type` | echoed question type |
| `probabilities` | map option-key → probability (sums to 1) |
| `choice` / `noul` / `score` + `label` / `confidence` | type-specific summary fields |

Option key order: `noul` prefers `[false, true]`; `choice` follows criteria key order (feature path sorts keys where needed for train compatibility); `score` uses levels `0..n-1`.

## Metric definitions

Per decision, prediction `p` and gold distribution `g` are aligned to canonical option order:

| metric | definition | over |
|---|---|---|
| accuracy | argmax(p) == gold label (`noul`: p_true ≥ 0.5 ⇒ "true") | all 2,000 |
| soft accuracy | Σ_k p_k g_k | noul + choice (1,200) |
| Brier | Σ_k (p_k − g_k)² | noul + choice |
| KL / TV | Σ_k g_k log(g_k / p_k) and ½ Σ_k \|p_k − g_k\| | noul + choice |
| ECE | 15 equal-width bins; confidence = max_k p_k | all 2,000 |
| NLL | −log p[gold label] | all 2,000 |
| score MAE / within-1 | \|Σ_i i·p_i − gold expected score\|, and whether ≤ 1 | score (800) |

Implementation: `velo_base/metrics.py`. Soft / Brier / KL / TV **exclude** score decisions.

## Calibration

| field | value |
|---|---|
| base transform | `clip_renorm` (clip negative head outputs at 0, renormalize) |
| temperature | per question type: choice 0.85, noul 0.95, score 0.85 |
| fit | OOS train holdout only |
| default mode | `calibrated` |
| companion | `clip` (clip_renorm, T = 1) — best sealed Brier under the hard floors |

Temperature scaling does not change argmax, so calibrated and clip modes share sealed accuracy (0.7535).
