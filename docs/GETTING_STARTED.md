# Getting started — Velo Base 1.0


## Git LFS

`checkpoints/hybrid/bundle.pkl` (~177 MB) is a Git LFS object. Install Git LFS, clone, then run `git lfs pull` before inference. GitHub’s auto-generated “Source code” zip/tar for a tag contains LFS **pointers**, not the weights — clone the repo (or use a release asset that bundles real weights) if you need a runnable tree.

## 1. Clone and install

```bash
git clone <repo-url> velo-base
cd velo-base
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Editable install (optional):

```bash
pip install -e .
```

## 2. Run inference on the sample case

```bash
python -m velo_base \
  --ckpt checkpoints/hybrid \
  --case-json examples/example_case.json \
  --mode calibrated
```

Equivalent:

```bash
python scripts/infer.py --ckpt checkpoints/hybrid --case-json examples/example_case.json
python examples/demo_infer.py
```

You should see JSON with keys such as `action`, `needs_review`, `outcome`, `risk`, `urgency`, each containing `probabilities` and a type-specific prediction.

## 3. Library use

```python
import json
from velo_base import load_bundle, predict

bundle = load_bundle("checkpoints/hybrid")
case = json.load(open("examples/example_case.json"))
out = predict(
    bundle,
    state=case["state"],
    questions=case["questions"],
    workflow=case["workflow"],
    mode="calibrated",  # or "clip"
)
print(out["action"]["choice"], out["action"]["confidence"])
```

## 4. Modes

| mode | meaning |
|---|---|
| `calibrated` | Default. `clip_renorm` + per-qtype temperature. Best ECE tradeoff. |
| `clip` | `clip_renorm`, T=1. Same argmax/accuracy; slightly better Brier. |
| `raw_soft_norm` | Legacy; not recommended for new work. |

## 5. Next steps

- Understand I/O: [`../PROTOCOL.md`](../PROTOCOL.md)
- Refit on your vertical: [`FINE_TUNING.md`](FINE_TUNING.md)
- Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Robustness results: [`ROBUSTNESS.md`](ROBUSTNESS.md)
