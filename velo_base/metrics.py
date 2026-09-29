# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""Typed-decision metrics matching the typed-decisions / Laya harness conventions.

Conventions (per decision):
- accuracy: argmax (noul: true iff p_true >= 0.5). All decisions.
- soft accuracy, Brier, KL, TV: against gold distribution, over noul+choice ONLY.
- score MAE / within-1: over score decisions only.
- NLL: -log p[gold] over all decisions.
- ECE: max-prob confidence vs argmax correctness, 15 equal-width bins, all decisions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


def ece_score(rows: Sequence[Tuple[int, Optional[np.ndarray]]], n_bins: int = 15) -> float:
    """ECE over (confidence, correctness) with 15 equal-width bins on (0, 1]."""
    confs, corrects = [], []
    for gold_idx, probs in rows:
        if probs is None:
            continue
        p = np.asarray(probs, dtype=np.float64)
        if p.sum() <= 0:
            continue
        p = p / p.sum()
        pred = int(p.argmax())
        confs.append(float(p[pred]))
        corrects.append(1.0 if pred == gold_idx else 0.0)
    if not confs:
        return float("nan")
    confs = np.asarray(confs)
    corrects = np.asarray(corrects)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if np.any(m):
            ece += (m.sum() / n) * abs(corrects[m].mean() - confs[m].mean())
    return float(ece)


def _noul_pred_idx(p: np.ndarray, keys: Optional[Sequence[str]], hard: int) -> int:
    p = np.asarray(p, dtype=np.float64)
    if keys and "true" in keys:
        ti = list(keys).index("true")
        if p[ti] >= 0.5:
            return ti
        # binary false/true
        if len(p) == 2:
            return 1 - ti
        return int(p.argmax())
    # fallback: index 1 is true
    if len(p) >= 2:
        return 1 if p[1] >= 0.5 else 0
    return int(p.argmax())


def summarize(
    hard_idx: Sequence[int],
    probs: Sequence[np.ndarray],
    soft_targets: Sequence[Optional[np.ndarray]] = (),
    gold_scores: Sequence[Optional[float]] = (),
    qtypes: Sequence[Optional[str]] = (),
    keys_list: Sequence[Optional[Sequence[str]]] = (),
) -> Dict[str, float]:
    """Summarize metrics. If qtypes provided, soft/Brier/KL/TV skip score (typed-decision protocol).
    Without qtypes, falls back to including all soft_targets (legacy).
    """
    rows = []
    soft_acc, brier, kl, tv = [], [], [], []
    nll = []
    mae, within1 = [], []
    use_typed_protocol = bool(qtypes) and len(qtypes) == len(hard_idx)

    for i, (hi, p) in enumerate(zip(hard_idx, probs)):
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, EPS, None)
        p = p / p.sum()
        qt = qtypes[i] if use_typed_protocol else None
        keys = keys_list[i] if keys_list and i < len(keys_list) else None

        if qt == "noul":
            pred = _noul_pred_idx(p, keys, int(hi))
        else:
            pred = int(p.argmax())
        rows.append((int(hi), p))
        nll.append(float(-np.log(p[int(hi)])))

        if soft_targets:
            gp = soft_targets[i]
            if gp is not None:
                gp = np.asarray(gp, dtype=np.float64)
                if gp.sum() > 0:
                    gp = gp / gp.sum()
                    n = min(len(p), len(gp))
                    pp, gg = p[:n], gp[:n]
                    # Soft metrics only on choice+noul (typed-decision protocol)
                    if (not use_typed_protocol) or (qt in ("choice", "noul")):
                        soft_acc.append(float((pp * gg).sum()))
                        brier.append(float(((pp - gg) ** 2).sum()))
                        tv.append(float(0.5 * np.abs(pp - gg).sum()))
                        gc = np.clip(gg, EPS, None)
                        kl.append(float((gg * np.log(gc / np.clip(pp, EPS, None))).sum()))

        if gold_scores and gold_scores[i] is not None:
            if (not use_typed_protocol) or qt == "score" or qt is None:
                exp = float(np.arange(len(p)).dot(p))
                err = abs(exp - float(gold_scores[i]))
                mae.append(err)
                within1.append(1.0 if err <= 1.0 else 0.0)

    # accuracy with noul rule when qtypes given
    correct = 0
    for i, (hi, p) in enumerate(rows):
        qt = qtypes[i] if use_typed_protocol else None
        keys = keys_list[i] if keys_list and i < len(keys_list) else None
        if qt == "noul":
            pred = _noul_pred_idx(p, keys, hi)
        else:
            pred = int(np.asarray(p).argmax())
        correct += int(pred == hi)

    out: Dict[str, float] = {
        "n": float(len(rows)),
        "accuracy": correct / max(len(rows), 1),
        "ece": ece_score(rows),
        "nll": float(np.mean(nll)) if nll else float("nan"),
    }
    if soft_acc:
        out["soft_accuracy"] = float(np.mean(soft_acc))
        out["brier"] = float(np.mean(brier))
        out["kl"] = float(np.mean(kl))
        out["tv"] = float(np.mean(tv))
    if mae:
        out["score_mae"] = float(np.mean(mae))
        out["within_1"] = float(np.mean(within1))
        out["within_1_level"] = out["within_1"]  # alias used by older seal JSON
    return out
