# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""Velo Base 1.0 — efficient typed-decision specialist (refittable base)."""

__version__ = "1.0.0"

from velo_base.infer import load_bundle, predict

__all__ = ["__version__", "load_bundle", "predict"]
