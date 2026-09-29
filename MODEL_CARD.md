# Model card — Velo Base 1.0

## Model details

| Field | Value |
|---|---|
| Name | Velo Base 1.0 |
| Package | `velo-base` / `velo_base` |
| Version | 1.0.0 |
| Author | Santhosh Reddy |
| License | Apache-2.0 |
| Architecture | Classical specialist: HistGradientBoosting soft heads + dict/state features + TF-IDF→TruncatedSVD(256) + optional CS distill blend + `clip_renorm` + per-qtype temperature |
| Neural net? | No |
| Approx. complexity | ~91 HGBR heads, ~898k tree nodes, ~7.7M SVD floats |
| Installed size | ~177 MB (`checkpoints/hybrid/bundle.pkl`) |
| Default mode | `calibrated` |

## Intended use

- Typed-decision inference on schemas shaped like JSON state → `choice` / `noul` / `score` questions.
- **Base** for refitting / fine-tuning heads onto vertical typed-decision problems (ops, marketing routing, finance-style triage, agent-tool checks) that share a similar I/O contract.
- Research on calibration, selective prediction, and presentation robustness under tight size/latency floors.

## Out of scope

- Open-ended language generation or general chat.
- Arbitrary domains without schema alignment or refit (including open-ended quant trading).
- Safety-critical deployment without human oversight and domain-specific validation.
- Claiming SOTA over all typed-decision systems from this release alone.

## Training data

- Dataset: [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), revision `ea9306458d6e9563628369a3d1e72e362fb381d2` (Apache-2.0).
- Same public dataset used by other typed-decision systems; rows are teacher-labelled and synthetic in character.
- Official train (1,200 cases) used for fitting and calibration holdout; official test (400 cases / 2,000 decisions) sealed after freeze.
- Dataset rows are **not** shipped in this repository.

## Evaluation data

- Official `test` split of the pinned dataset revision (sealed once).
- First-party presentation / alien / ultra-fresh stress suites (see `docs/ROBUSTNESS.md`). These transform presentation only; gold remains official teacher labels.

## Metrics (sealed, calibrated)

| Metric | Value |
|---|---:|
| accuracy | 0.7535 |
| soft accuracy | ~0.557 |
| Brier | ~0.072 (clip-best ~0.069) |
| ECE | ~0.120 |
| NLL | ~0.647 |
| score MAE | ~0.265 |
| within-1 | 0.986 |
| top-50% selective | 0.894 |
| p50 ms / 5-decision case | ~32 |
| size MB | ~177 |

Published accuracy context (not re-run by us on the ultra-fresh suite): Laya ~0.766 (~98.4% of Laya); TypeSafe Jev 1.13 ~0.727 (~103.6% of Jev). Laya p50 latency measured on this eval host: ~740 ms. See `BENCHMARK.md`.

## Limitations

- Synthetic / teacher-labelled benchmark; transfer to messy production records is not automatic.
- Brier is the weakest soft metric relative to published specialist figures; clip mode improves Brier slightly (~0.069) at the cost of ECE.
- Presentation stretch bar (≥72% on all transforms) fails on Title-Case, CS shoutcase padding, coding noise, and hard combos — must-bar ≥60% still passes.
- Alien (unmapped) keys measure feature-path brittleness; packaging helps but does not invent semantics for truly novel fields.
- Refitting on small vertical sets can overfit; hold out a domain eval set.

## Ethical considerations

Decisions that affect people (fraud flags, access, credit-like triage) need human review, domain policy, and monitoring. This model outputs distributions, not authorizations.

## Citation

Copyright 2026 Santhosh Reddy. See `LICENSE` and `NOTICE`. Credit `LocalLLaMA/typed-decisions` when publishing derivative results.
