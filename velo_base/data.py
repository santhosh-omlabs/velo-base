# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""Load LocalLLaMA/typed-decisions and flatten into question examples."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from velo_base.rich_features import state_feature_dict
from velo_base.schema import QuestionExample, parse_row

DATASET_NAME = "LocalLLaMA/typed-decisions"
# Revision used when training and sealing Velo Base 1.0.
DATASET_REVISION = "ea9306458d6e9563628369a3d1e72e362fb381d2"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "typed_decisions"


def load_splits(cache_dir: str | Path | None = None, revision: str = DATASET_REVISION):
    """Load train/test splits, preferring a local cache under ``data/typed_decisions``."""
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    if cache.exists():
        from datasets import load_from_disk

        return load_from_disk(str(cache))
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, "all", revision=revision)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache))
    return ds


def examples_from_split(ds, split: str) -> List[QuestionExample]:
    items: List[QuestionExample] = []
    for row in ds[split]:
        items.extend(parse_row(row))
    return items


def cases_from_split(ds, split: str) -> List[dict]:
    from velo_base.schema import _as_obj

    cases = []
    for row in ds[split]:
        cases.append(
            {
                "id": row["id"],
                "workflow": row["workflow"],
                "state": _as_obj(row["state"]),
                "questions": _as_obj(row["questions"]),
                "gold": _as_obj(row["gold"]),
            }
        )
    return cases


def build_items(ds, split: str) -> List[Dict[str, Any]]:
    """Flatten a split into per-decision feature dicts used by eval and training."""
    items: List[Dict[str, Any]] = []
    for row in ds[split]:
        for ex in parse_row(row):
            feat = state_feature_dict(ex.state_obj, ex.workflow, ex.qid, ex.qtype)
            text = ex.state_text + " " + ex.instructions
            items.append(
                {
                    "feat": feat,
                    "text": text,
                    "soft": ex.soft,
                    "hard": ex.hard,
                    "keys": ex.keys,
                    "slot": f"{ex.workflow}::{ex.qid}",
                    "gold_score": ex.gold_score,
                    "qtype": ex.qtype,
                    "workflow": ex.workflow,
                    "qid": ex.qid,
                    "case_id": ex.case_id,
                }
            )
    return items
