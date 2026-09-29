# BENCHMARK — Velo Base 1.0

Source of truth for sealed numbers: [`artifacts/seal_v1.json`](artifacts/seal_v1.json) (and `checkpoints/hybrid/meta.json`). Dataset: Hugging Face `LocalLLaMA/typed-decisions`, revision `ea9306458d6e9563628369a3d1e72e362fb381d2`. Official seal: **400 cases / 2,000 decisions**.

Metric protocol (typed-decision / Laya conventions): soft accuracy, Brier, KL, and TV on **choice + noul only**; accuracy / NLL / ECE on all decisions; MAE / within-1 on score. See [`PROTOCOL.md`](PROTOCOL.md).

---

## A. Our sealed metrics

| Metric | Calibrated (default) | Clip (Brier companion) |
|---|---:|---:|
| Accuracy | **0.7535** | 0.7535 |
| Soft accuracy | ~0.557 | ~0.547 |
| Brier | ~0.072 | ~0.069 |
| ECE | ~0.120 | ~0.148 |
| NLL | ~0.647 | ~0.675 |
| Score MAE | ~0.265 | ~0.268 |
| within-1 | 0.986 | 0.985 |
| Top-50% selective | 0.894 | — |
| p50 ms / 5-decision case | **~32** | ~32 |
| Package size | **~177 MB** | ~177 MB |

Hard floors: accuracy ≥ 0.752 · p50 ≤ 35 ms · size ≤ 200 MB — **all PASS**.

By workflow (calibrated, accuracy): agent_trace_observability 0.726 · customer_service ~0.71 · invoice_processing 0.844 · security_incidents ~0.73.

Latency measured on our eval host (`artifacts/hardware.json`: 8-thread Xeon-class Linux). Do not treat ~32 ms as a universal hardware claim.

---

## B. Published accuracy context (Laya / Jev)

We cite **published / known** sealed accuracies for orientation. We did **not** locally re-test Laya or Jev head-to-head on the ultra-fresh robustness suites used for Velo packaging stress.

| System | Accuracy (source) | Velo relative |
|---|---:|---|
| **Velo Base 1.0** (ours, sealed) | **0.7535** | — |
| Laya | ~0.766 (published / model-card; harness-reproduced soft/Brier differ — see seal references) | Velo ≈ **98.4%** of Laya accuracy |
| TypeSafe Jev 1.13 | ~0.727 (published, generalist) | Velo ≈ **103.6%** of Jev accuracy |

**Latency (same-host measurement):** Laya p50 ≈ **740 ms** per 5-decision-style case was measured on this eval host (`artifacts/laya_latency.json`). Velo p50 ≈ **32 ms** on the same machine class. That is the speed story we can defend with a local measurement. We do not invent hardware comparisons beyond that.

Jev is a published generalist; specialists and generalists are not directly comparable (as the dataset card notes). Figures are for context, not a claim that we re-sealed those systems in this repository.

---

## C. First-party robustness (Velo-only)

Packaging/normalize fixes (sorted choice keys, `prepare_state` flatten/unwrap, aliases, opaque remapping, date/unicode) were applied **without retraining** sealed heads.

Ultra-fresh suite (~2000-decision buckets; finance/marketing/coding/research mess; varied dates, ALL CAPS, alien keys; seed `20260929`):

| Gate (ours) | Result |
|---|:---:|
| Presentation all ≥ **60%** (must) | **PASS** |
| Presentation all ≥ **72%** (stretch) | **FAIL** |
| Sealed clean ≥ 0.752 | **PASS** |
| Alien macro accuracy | **~0.770** |

Stretch failures (honest): Title-Case, CS shoutcase padding, coding-agent noise, combo_hard, combo_ultra — all still ≥60%. Full bucket table: [`docs/ROBUSTNESS.md`](docs/ROBUSTNESS.md) · [`artifacts/robustness_ultra_fresh.json`](artifacts/robustness_ultra_fresh.json).

**Explicit:** Laya and Jev were **not** re-run on this ultra-fresh suite for the public comparison tables.

---

## Novelty framing (for readers)

The interesting claim is **extreme efficiency under quality floors**: a small classical stack, CPU-tens-of-ms, ≤200 MB, retaining ~98% of Laya-class sealed accuracy, practical as a **refittable base** for vertical typed-decision schemas. Trees + SVD + calibration after a small neural attempt failed those floors. Not a SOTA foundation model; not a new theory of decisions — a serious specialist artifact with honest limits.
