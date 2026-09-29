# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""Public inference helpers for the Velo Base 1.0 hybrid (HGBR + SVD) checkpoint.

Default mode is ``calibrated`` (clip_renorm + per-qtype temperature fitted on
an OOS train holdout). Inputs are ``state`` and ``questions`` only; gold
``factors`` are never read. States are passed through ``prepare_state`` so
presentation noise (wrappers, aliases, casing, date/unicode forms) is
normalized without retraining the sealed heads.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np

from velo_base.normalize import prepare_state, sanitize_instructions
from velo_base.rich_features import state_feature_dict
from velo_base.schema import _as_obj, criteria_to_options, state_to_text

EPS = 1e-12
PathLike = Union[str, Path]


def clip_renorm(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    s = np.clip(s, 0, None)
    total = s.sum()
    if total <= 0:
        return np.ones_like(s) / len(s)
    return s / total


def temp_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    if abs(t - 1.0) < 1e-12:
        return probs
    logit = np.log(np.clip(probs, EPS, 1.0)) / t
    logit = logit - logit.max()
    e = np.exp(logit)
    return e / e.sum()


def load_bundle(ckpt: PathLike) -> dict:
    """Load ``bundle.pkl`` from a checkpoint directory."""
    path = Path(ckpt)
    bundle_path = path / "bundle.pkl" if path.is_dir() else path
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


def predict(
    bundle: Mapping[str, Any],
    state: Any,
    questions: Any,
    workflow: str,
    mode: str = "calibrated",
) -> Dict[str, dict]:
    """Predict one answer distribution per question.

    Parameters
    ----------
    bundle:
        Loaded checkpoint dict (``load_bundle``).
    state, questions:
        Case fields as JSON objects or JSON strings.
    workflow:
        One of the four typed-decisions workflows, or a vertical schema id
        after you refit heads for a custom domain.
    mode:
        ``calibrated`` (default), ``clip``, or ``raw_soft_norm`` (legacy).
    """
    if mode not in ("calibrated", "clip", "raw_soft_norm"):
        raise ValueError(f"unknown mode: {mode!r}")

    state_obj = prepare_state(_as_obj(state))
    state_text = state_to_text(state_obj)
    qs = _as_obj(questions)
    dv, tfidf, svd = bundle["dict_vectorizer"], bundle["tfidf"], bundle["svd"]
    cal = bundle.get("calibration") or {}
    t_by = cal.get("T_by_qtype") or {"choice": 1.0, "noul": 1.0, "score": 1.0}
    weights = bundle.get("per_wf_weights") or {
        "cs": 0.6,
        "sec": 0.0,
        "agent": 1.0,
        "inv": 1.0,
    }
    orig = bundle["hgb_models"]
    big_seeds = bundle.get("big_models_seeds") or []
    cs_seeds = bundle.get("cs_models_seeds") or []
    cs_dist = bundle.get("cs_distill")

    def cs_z(state_json: str):
        if not cs_dist:
            return None
        vec, cats, nums = cs_dist["vec"], cs_dist["cats"], cs_dist["nums"]
        xv = vec.transform([state_json])
        feats = []
        for _k, (_le, clf) in cats.items():
            feats.extend(list(clf.predict_proba(xv)[0]))
        for _k, reg in nums.items():
            feats.append(float(reg.predict(xv)[0]))
        return np.asarray(feats, np.float32)

    st_json = json.dumps(state_obj, ensure_ascii=False)
    answers: Dict[str, dict] = {}
    for qid, qd in qs.items():
        qtype = qd["type"]
        keys, _opts = criteria_to_options(qtype, qd.get("criteria"))
        slot = f"{workflow}::{qid}"
        feat = state_feature_dict(state_obj, workflow, qid, qtype)
        text = state_text + " " + sanitize_instructions(str(qd.get("instructions", "")))
        xd = dv.transform([feat])
        xt = svd.transform(tfidf.transform([text]))
        x = np.hstack([xd, xt]).astype(np.float32)

        def pred_regs(models, x_mat=x):
            regs = models[slot]
            return np.array([r.predict(x_mat)[0] for r in regs], dtype=np.float64)

        if workflow in ("invoice_processing", "agent_trace_observability"):
            raw = pred_regs(orig)
        elif workflow == "security_incidents":
            raw = np.mean([pred_regs(m) for m in big_seeds], axis=0)
        else:
            o = pred_regs(orig)
            z = cs_z(st_json)
            x2 = np.hstack([x, z.reshape(1, -1)]) if z is not None else x

            def pred_cs(models):
                regs = models[slot]
                return np.array([r.predict(x2)[0] for r in regs], dtype=np.float64)

            c = np.mean([pred_cs(m) for m in cs_seeds], axis=0)
            raw = weights.get("cs", 0.6) * o + (1 - weights.get("cs", 0.6)) * c

        if mode == "raw_soft_norm":
            s = raw - raw.min()
            p = s / s.sum() if s.sum() > 0 else np.ones_like(s) / len(s)
        else:
            p = clip_renorm(raw)
            if mode == "calibrated":
                p = temp_scale(p, t_by.get(qtype, 1.0))

        pred_i = int(p.argmax())
        ans: Dict[str, Any] = {
            "type": qtype,
            "probabilities": {k: float(p[i]) for i, k in enumerate(keys)},
        }
        if qtype == "choice":
            ans["choice"] = keys[pred_i]
            ans["confidence"] = float(p[pred_i])
        elif qtype == "noul":
            true_i = keys.index("true") if "true" in keys else pred_i
            ans["noul"] = float(p[true_i])
            if "true" in keys:
                ans["label"] = "true" if p[true_i] >= 0.5 else "false"
            else:
                ans["label"] = keys[pred_i]
            ans["confidence"] = float(p.max())
        else:
            ans["score"] = float(np.arange(len(p)).dot(p))
            ans["label"] = keys[pred_i]
            ans["confidence"] = float(p[pred_i])
        answers[qid] = ans
    return answers
