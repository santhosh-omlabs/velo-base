# Reproducibility — Velo Base 1.0

## 1. Environment

- Python ≥ 3.10
- Dependencies: `requirements.txt` / `pyproject.toml`
- PyTorch is **not** required to load or run `checkpoints/hybrid`
- Eval host used for sealing latency: see `artifacts/hardware.json` (8 CPUs, Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## 2. Checkpoint

| path | role |
|---|---|
| `checkpoints/hybrid/bundle.pkl` | Production weights (~177 MB) |
| `checkpoints/hybrid/meta.json` | Modes, sealed summary, calibration |

Default mode: `calibrated`.

## 3. Dataset

```text
name:     LocalLLaMA/typed-decisions
revision: ea9306458d6e9563628369a3d1e72e362fb381d2
```

`velo_base.data.load_splits()` downloads that revision on first use and caches under `data/typed_decisions/` (gitignored). Official test: 400 cases / 2,000 decisions. Protocol: `PROTOCOL.md`.

## 4. Smoke inference

```bash
python -m velo_base --ckpt checkpoints/hybrid --case-json examples/example_case.json
# or
python examples/demo_infer.py
```

Expect a JSON object with per-question `probabilities` and type-specific fields. A checked-in example prediction is `examples/example_pred.json` (illustrative; floating noise across sklearn versions may differ slightly).

## 5. Expected sealed numbers (calibrated)

| metric | expected |
|---|---:|
| accuracy | 0.7535 |
| soft accuracy | ~0.557 |
| Brier | ~0.072 |
| ECE | ~0.120 |
| NLL | ~0.647 |
| score MAE | ~0.265 |
| top-50% selective | 0.894 |
| p50 ms / case | ~32 |
| size MB | ~177 |

Clip companion: same accuracy; Brier ~0.069; ECE higher (~0.148).

Hard floors: accuracy ≥ 0.752 · p50 ≤ 35 · size ≤ 200 — PASS at seal time.

## 6. Robustness artifacts

- `artifacts/robustness_ultra_fresh.json` — Velo-only presentation / alien buckets
- Methodology and narrative: `docs/ROBUSTNESS.md`

Presentation must-bar ≥60%: PASS. Stretch ≥72%: FAIL on documented modes. Alien macro ~0.770.

## 7. What is not reproduced in this tree

- Full training from scratch (multi-hour; large intermediate checkpoints omitted)
- Head-to-head re-runs of Laya / Jev on the ultra-fresh suite
- Multi-GB experimental checkpoints

Published Laya/Jev accuracies are cited for context only (`BENCHMARK.md`).
