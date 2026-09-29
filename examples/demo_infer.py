#!/usr/bin/env python3
# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0
"""Minimal library demo: load Velo Base 1.0 and predict on the sample case."""
from __future__ import annotations

import json
from pathlib import Path

from velo_base import load_bundle, predict

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bundle = load_bundle(ROOT / "checkpoints" / "hybrid")
    case = json.loads((ROOT / "examples" / "example_case.json").read_text())
    answers = predict(
        bundle,
        case["state"],
        case["questions"],
        case["workflow"],
        mode="calibrated",
    )
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
