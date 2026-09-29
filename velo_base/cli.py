# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: ``velo-infer`` / ``python -m velo_base``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from velo_base.infer import load_bundle, predict


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Velo Base 1.0 single-case inference")
    ap.add_argument("--ckpt", default="checkpoints/hybrid", help="Checkpoint directory")
    ap.add_argument("--case-json", required=True, help="Path to a case JSON file")
    ap.add_argument(
        "--mode",
        choices=["calibrated", "clip", "raw_soft_norm"],
        default="calibrated",
    )
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        # Prefer CWD-relative; fall back to package-repo root heuristic
        cand = Path.cwd() / ckpt
        ckpt = cand if cand.exists() else Path(__file__).resolve().parents[1] / args.ckpt

    bundle = load_bundle(ckpt)
    case = json.loads(Path(args.case_json).read_text())
    answers = predict(
        bundle,
        case["state"],
        case["questions"],
        case["workflow"],
        mode=args.mode,
    )
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
