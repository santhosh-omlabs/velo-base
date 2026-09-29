# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""Typed-decision case parsing (choice / score / noul)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from velo_base.normalize import (
    has_opaque_residue,
    leaf_values_text,
    prepare_state,
    sanitize_instructions,
)


WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)

QUESTION_TYPES = ("choice", "score", "noul")


@dataclass
class QuestionExample:
    case_id: str
    workflow: str
    qid: str
    qtype: str
    state_text: str
    state_obj: Any
    instructions: str
    keys: List[str]
    option_texts: List[str]
    soft: List[float]
    hard: int
    gold_score: Optional[float]
    gold_label: str


def _as_obj(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def state_to_text(state: Any) -> str:
    """Train-compatible text view: sorted JSON (sealed TF-IDF vocabulary).

    When keys remain opaque after prepare_state, append leaf values so TF-IDF
    still sees value tokens (must not alter the JSON prefix on clean inputs).
    """
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            return state
    base = json.dumps(state, sort_keys=True, ensure_ascii=False)
    if has_opaque_residue(state):
        extra = leaf_values_text(state)
        if extra:
            return base + " " + extra
    return base


def criteria_to_options(qtype: str, criteria: Any) -> Tuple[List[str], List[str]]:
    if qtype == "score":
        opts = [str(c) for c in (criteria or [])]
        keys = [str(i) for i in range(len(opts))]
        return keys, opts
    if qtype == "noul":
        if not criteria:
            return ["false", "true"], ["false", "true"]
        keys = list(criteria.keys())
        return keys, [str(criteria[k]) for k in keys]
    # choice — canonicalize option order so position-indexed heads are
    # invariant to criteria presentation order (train already uses sorted keys).
    keys = sorted(criteria.keys(), key=lambda k: str(k))
    return keys, [str(criteria[k]) for k in keys]


def soft_from_gold(keys: Sequence[str], gold: Dict[str, Any]) -> Tuple[List[float], int, Optional[float], str]:
    label = str(gold["label"])
    probs = gold.get("probabilities") or {}
    soft = [float(probs.get(k, 0.0)) for k in keys]
    if sum(soft) <= 0:
        soft = [1.0 if k == label else 0.0 for k in keys]
    s = sum(soft) or 1.0
    soft = [p / s for p in soft]
    if label in keys:
        hard = keys.index(label)
    else:
        hard = int(max(range(len(soft)), key=lambda i: soft[i]))
    gold_score = float(gold["score"]) if "score" in gold else None
    return soft, hard, gold_score, label


def parse_row(row: Dict[str, Any]) -> List[QuestionExample]:
    qs = _as_obj(row["questions"])
    gold = _as_obj(row["gold"])
    state_obj = prepare_state(_as_obj(row["state"]))
    state_text = state_to_text(state_obj)
    out: List[QuestionExample] = []
    for qid, qd in qs.items():
        qtype = qd["type"]
        keys, opts = criteria_to_options(qtype, qd.get("criteria"))
        soft, hard, gold_score, label = soft_from_gold(keys, gold[qid])
        out.append(
            QuestionExample(
                case_id=str(row["id"]),
                workflow=str(row["workflow"]),
                qid=str(qid),
                qtype=qtype,
                state_text=state_text,
                state_obj=state_obj,
                instructions=sanitize_instructions(str(qd.get("instructions", ""))),
                keys=keys,
                option_texts=opts,
                soft=soft,
                hard=hard,
                gold_score=gold_score,
                gold_label=label,
            )
        )
    return out


def question_slot_id(workflow: str, qid: str) -> str:
    return f"{workflow}::{qid}"
