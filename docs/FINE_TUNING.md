# Fine-tuning / refitting — Velo Base 1.0

Velo Base 1.0 is positioned as a **base specialist**: sealed on the public typed-decisions schemas, then meant to be **refit** onto vertical problems that share the same shape — JSON/state in, typed `choice` / `noul` / `score` out.

This document is practical guidance. The release ships production inference weights; full training scripts from the research tree are not required to *use* the base, but the recipes below match how the sealed model was built.

## 1. When refit makes sense

Good fit:

- Ops / incident triage, marketing routing, finance-style disposition, agent-tool allow/deny/review — if you can write a schema of questions with finite criteria.
- You have hundreds to thousands of labelled cases (or strong soft labels from a teacher).
- Latency and size budgets still matter (CPU, ≤200 MB class).

Poor fit:

- Open-ended generation, free-form RAG answers, or domains with no stable JSON schema.
- Expecting zero-shot transfer to unrelated fields without any labels.

## 2. Label format

Mirror the typed-decisions case shape:

```json
{
  "id": "vertical_000001",
  "workflow": "your_vertical_id",
  "state": { "...": "business payload" },
  "questions": {
    "disposition": {
      "type": "choice",
      "instructions": "Select the best next action.",
      "criteria": {
        "approve": "Approve as-is.",
        "reject": "Reject.",
        "escalate": "Send to a human."
      }
    },
    "needs_human": {
      "type": "noul",
      "instructions": "A human should review this case.",
      "criteria": { "false": "...", "true": "..." }
    },
    "urgency": {
      "type": "score",
      "instructions": "Urgency from 0 (low) to 4 (critical).",
      "criteria": ["low", "elevated", "high", "severe", "critical"]
    }
  },
  "gold": {
    "disposition": { "soft": {"approve": 0.1, "reject": 0.1, "escalate": 0.8}, "hard": "escalate" },
    "needs_human": { "soft": {"false": 0.2, "true": 0.8}, "hard": "true" },
    "urgency": { "soft": [0.0, 0.1, 0.2, 0.5, 0.2], "hard": 3 }
  }
}
```

Notes:

- Prefer **soft** distributions when you have a teacher or annotator disagreement; hard labels alone are fine but train a sharper head.
- Keep option keys stable across the dataset; sorting at feature time is already handled for choice keys in packaging.
- Run states through the same `prepare_state` path used at inference so train/serve agree on aliases and wrappers.

## 3. What to freeze vs refit

| Component | Recommendation |
|---|---|
| TF-IDF + SVD tables | Freeze initially if vertical text looks similar; refit if vocabulary shifts hard |
| Dict / state feature extractors | Extend heuristics for new fields; keep canonicalization (`normalize.py`) |
| HGBR soft heads | **Refit** per new `workflow::qid` slots |
| CS distill blend | Optional; only if you rebuild a similar auxiliary signal |
| Temperature / `clip_renorm` | Re-fit temperatures on an OOS holdout of *your* vertical; never on the final test set |

Starting point that usually works: freeze global text tables, refit heads on vertical labels, then calibrate T on a holdout.

## 4. How many samples?

Rough guidance (not a theorem):

| Regime | Expectation |
|---|---|
| < 100 cases | High overfit risk; prefer stronger soft labels + heavy regularization / shallower trees |
| ~300–1,000 | Often enough to beat a frozen base on in-domain holdout if schema is tight |
| ≥ 1,200 (train-scale) | Comfortable for multi-question workflows similar to the public benchmark |

Always keep a **held-out** vertical eval set (case-level, not question-shuffled) and report accuracy, soft, Brier/ECE, and a presentation-stress smoke (casing, wrappers) before shipping.

## 5. Eval protocol for verticals

1. Freeze a candidate; evaluate **once** on the holdout.
2. Report the same metric split as `PROTOCOL.md` (soft/Brier on choice+noul only).
3. Add selective accuracy at 50% coverage if you will abstain in production.
4. Run a small presentation stress (ALL CAPS, Title-Case, nested wrappers) — packaging should keep ≥60% if keys are canonicalized.

## 6. Domain-shift risks

- New field names that never appear in synonyms → treat as alien keys; extend `normalize.py` aliases deliberately, then re-stress.
- Different option cardinalities → new heads; do not force old slot weights.
- Label policy drift (teacher vs human) → soft targets help, but document whose gold you use.
- **Refit ≠ overfitting:** improving train loss while holdout Brier/ECE collapse is failure. Prefer early stopping on holdout soft/Brier, not train accuracy alone.

## 7. Practical pitfalls

- Feeding gold `factors` into features at train time and forgetting them at serve time (train/serve skew).
- Forgetting `prepare_state` in a custom training loop.
- Calibrating temperatures on the same set you report as “test.”
- Shipping `raw_soft_norm` probabilities into a Brier-sensitive consumer.

## 8. Suggested workflow checklist

1. Define schema (`workflow`, question ids, criteria).
2. Collect / teacher-label cases; split case-level train / holdout / test.
3. Freeze or lightly refresh SVD; refit HGBR heads.
4. Fit per-qtype temperatures on holdout.
5. Seal once on test; run presentation smoke.
6. Document domain limits in your fork’s model card.
