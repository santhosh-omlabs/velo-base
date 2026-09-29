# Architecture — Velo Base 1.0

## Overview

Velo Base 1.0 is a **classical** typed-decision specialist. It is not a neural language model. After a small neural attempt failed hard size and latency floors, the production stack settled on HistGradientBoosting soft heads over engineered state features plus a compressed text table.

```text
JSON state + questions
        │
        ▼
 prepare_state / canonicalize  (wrappers, aliases, casing, dates)
        │
        ├─► dict / heuristic state features ──► DictVectorizer
        │                                            │
        └─► state text + instructions ──► TF-IDF ──► SVD(256)
                                                     │
                                                     ▼
                                      concat feature row (float32)
                                                     │
                                                     ▼
                         per-slot HistGradientBoostingRegressor heads
                         (optional CS distill blend on customer_service)
                                                     │
                                                     ▼
                              clip_renorm  (+ per-qtype temperature)
                                                     │
                                                     ▼
                                         probabilities + summaries
```

## Components

| Piece | Role |
|---|---|
| `velo_base.normalize` | Presentation robustness: unwrap envelopes, synonym/alias maps, opaque remapping, date/unicode cleanup, choice-key sorting helpers |
| `velo_base.rich_features` | Flatten state + workflow heuristics into a dict feature row |
| `velo_base.schema` | Parse cases, criteria → option keys, train-compatible `state_to_text` |
| DictVectorizer + TF-IDF + TruncatedSVD(256) | Fitted at train time; stored in `bundle.pkl` |
| HGBR heads | Soft scores per option for each `workflow::qid` slot (~91 heads) |
| CS distill (optional) | Extra features + seed ensemble blend for customer_service |
| `clip_renorm` + T | Probabilityization and calibration without changing argmax when T is applied |

## Complexity honesty

| Quantity | Approx. |
|---|---:|
| HGBR heads / slots | ~91 |
| Tree nodes | ~898k |
| SVD-related floats | ~7.7M |
| Installed `bundle.pkl` | ~177 MB |
| Gzip size (not required to ship) | ~74 MB |

There is no meaningful “N million neural params” claim. Footprint is sklearn objects and feature tables.

## Why this stack

Hard floors for the release: sealed accuracy ≥ 0.752, warm p50 ≤ 35 ms per 5-decision case, package ≤ 200 MB. Trees + SVD + calibration cleared those floors while retaining ~98% of Laya-class published accuracy. That is the design thesis: **efficiency under quality floors**, then ship as a refittable base.

## Inference packaging

`predict()` always runs `prepare_state` before featurization. Packaging fixes (sorted choice keys, flatten/unwrap, aliases, opaque remapping, date/unicode) do **not** require weight refits when train already used snake_case and sorted choice keys. See `docs/ROBUSTNESS.md`.
