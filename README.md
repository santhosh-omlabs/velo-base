# Velo Base 1.0

**Velo** is a compact, CPU-fast typed-decision specialist: given a JSON state and typed questions (`choice`, `noul`, `score`), it returns a calibrated probability distribution per question. Version **1.0** is released as a **base** — sealed on Hugging Face [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), and intended for **refitting / fine-tuning** onto vertical schemas with a similar JSON/state → choice/score shape (ops triage, marketing routing, finance-style disposition, agent-tool checks).

It is **not** a general LLM foundation, not a neural net, and not a claim of new decision theory. It is an engineering artifact: quality floors under hard size and latency budgets, with honest specialist positioning and packaging robustness.

License: [Apache-2.0](LICENSE). Author: Santhosh Reddy.

## Honesty box

| Claim | Status |
|---|---|
| Sealed accuracy **0.7535** on the official typed-decisions test (400 cases / 2,000 decisions) | Measured here; see [`artifacts/seal_v1.json`](artifacts/seal_v1.json) |
| ~**98.4%** of published Laya accuracy (0.766); ~**103.6%** of TypeSafe Jev 1.13 (0.727) | Accuracy context from published/known figures — we did **not** re-run Laya/Jev on our ultra-fresh robustness suite |
| p50 latency **~32 ms** per 5-decision case on our eval host | Measured; Laya p50 on the **same host** was ~740 ms |
| Installed weights **~177 MB** | `checkpoints/hybrid/bundle.pkl` |
| Classical stack (HistGradientBoosting + TF-IDF→SVD), not a neural foundation | Intentional |
| Presentation-stress must-bar ≥60% | **PASS**; stretch ≥72% **FAIL** on some modes (documented) |

## Sealed metrics (calibrated default)

| Metric | Velo Base 1.0 | Notes |
|---|---:|---|
| Accuracy | **0.7535** | Official test |
| Soft accuracy | ~0.557 | choice + noul |
| Brier | ~0.072 (clip-best ~0.069) | Weakest relative soft metric; stated plainly |
| ECE | ~0.120 | |
| NLL | ~0.647 | |
| Score MAE | ~0.265 | |
| Top-50% selective | ~0.894 | |
| p50 latency | **~32 ms** | 5-decision case, CPU, 8 threads |
| Package size | **~177 MB** | |

Hard floors for this release: accuracy ≥ 0.752, p50 ≤ 35 ms, size ≤ 200 MB — all met.

Full tables: [`BENCHMARK.md`](BENCHMARK.md) · [`benchmarks/benchmarks.html`](benchmarks/benchmarks.html) · [`MODEL_CARD.md`](MODEL_CARD.md). Robustness: [`docs/ROBUSTNESS.md`](docs/ROBUSTNESS.md).

## What it is / what it is not

**Is:** a classical specialist — ~91 HistGradientBoosting soft heads, dict/state features, TF-IDF → truncated SVD (256), optional customer-service distill blend, `clip_renorm` + per-qtype temperature. Complexity is on the order of ~898k tree nodes and ~7.7M SVD floats; installed size ~177 MB. Trained supervised on `LocalLLaMA/typed-decisions` (revision pinned in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).

**Is not:** an open-ended LLM; a quant trading engine; a claim of SOTA over all typed-decision systems; a guarantee of transfer outside schemas that look like the training workflows.

The point of shipping a **base**: keep the feature + calibration stack, then **refit heads** on your vertical labels. See [`docs/FINE_TUNING.md`](docs/FINE_TUNING.md).

## Quickstart

```bash
# Requires Git LFS (the ~177 MB checkpoint is an LFS object)
#   macOS: brew install git-lfs && git lfs install
#   Debian/Ubuntu: sudo apt-get install git-lfs && git lfs install
git clone https://github.com/santhosh-omlabs/velo-base.git
cd velo-base
git lfs pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .
export PYTHONPATH=$PWD

python -m velo_base \
  --ckpt checkpoints/hybrid \
  --case-json examples/example_case.json \
  --mode calibrated
```

Library API:

```python
from velo_base import load_bundle, predict
import json

bundle = load_bundle("checkpoints/hybrid")
case = json.load(open("examples/example_case.json"))
answers = predict(bundle, case["state"], case["questions"], case["workflow"], mode="calibrated")
```

Runtime needs NumPy and scikit-learn for inference. Hugging Face `datasets` is needed only to download the benchmark for sealed eval. PyTorch is **not** required for the production checkpoint.

More detail: [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

## Architecture (one paragraph)

Trees won after a small neural attempt failed the size/latency floors. Soft heads are HistGradientBoosting regressors over a concatenated dict-feature vector and a TF-IDF→SVD(256) text table, with a light CS distill blend on the customer-service workflow, then `clip_renorm` and per-qtype temperature. Inference runs `prepare_state` first so wrappers, aliases, casing, and date/unicode presentation noise are normalized without weight refits. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Docs map

| Doc | Contents |
|---|---|
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | Clone → install → infer |
| [`docs/FINE_TUNING.md`](docs/FINE_TUNING.md) | Refit heads on vertical labels |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Stack detail |
| [`docs/ROBUSTNESS.md`](docs/ROBUSTNESS.md) | Presentation / alien stress (Velo-only) |
| [`PROTOCOL.md`](PROTOCOL.md) | I/O + metric contract |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Pins, commands, expected numbers |
| [`BENCHMARK.md`](BENCHMARK.md) | Sealed metrics + published context |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, limits |

## Citation / license

Copyright 2026 Santhosh Reddy. Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Please credit the benchmark dataset: Hugging Face `LocalLLaMA/typed-decisions` (Apache-2.0), revision `ea9306458d6e9563628369a3d1e72e362fb381d2`.
