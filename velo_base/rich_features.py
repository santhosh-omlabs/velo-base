# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""Rich state feature extraction (JSON flatten + workflow heuristics + text cues)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

import numpy as np

from velo_base.normalize import canonicalize_key, dig, prepare_state

MONEY_RE = re.compile(r"\$?\d[\d,]*\.?\d*")
LEAVE_RE = re.compile(
    r"\b(cancel|close my account|unsubscribe|switch to (a )?competitor|leaving|churn)\b",
    re.I,
)
URGENT_RE = re.compile(r"\b(asap|immediately|urgent|today|deadline|overdue)\b", re.I)
ANGRY_RE = re.compile(
    r"\b(angry|frustrated|unacceptable|lawsuit|lawyer|furious|ridiculous)\b", re.I
)
BENIGN_RE = re.compile(
    r"\b(scheduled maintenance|expected|known change|false positive|benign|authorized)\b",
    re.I,
)



def _num(v: Any, default: float = 0.0) -> float:
    """Coerce ints/floats/numeric strings (incl. $ and commas) for heuristics."""
    if v is None or v is False:
        return default
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace("$", "").replace(",", "").strip()
        if not s:
            return default
        # scientific notation / spaced strings
        try:
            return float(s)
        except Exception:
            return default
    try:
        return float(v)
    except Exception:
        return default


def flatten_dict(
    obj: Any,
    prefix: str = "",
    out: Dict[str, float] | None = None,
    depth: int = 0,
    also_leaf: bool = True,
):
    """Flatten JSON to dotted feature keys.

    Also emits leaf-only aliases (``leaf:<name>``) and last-two-segment aliases so
    residual nesting still overlaps the sealed DictVectorizer vocabulary when the
    full training path cannot be recovered.
    """
    if out is None:
        out = {}
    if depth > 8:
        return out
    if isinstance(obj, dict):
        out[prefix + "__len"] = float(len(obj))
        for k, v in obj.items():
            ck = canonicalize_key(k) if isinstance(k, str) else str(k)
            flatten_dict(v, f"{prefix}{ck}.", out, depth + 1, also_leaf=also_leaf)
    elif isinstance(obj, list):
        out[prefix + "__len"] = float(len(obj))
        nums = [_num(x) for x in obj if isinstance(x, (int, float, str)) and not isinstance(x, bool)]
        nums = [x for x in nums if True]
        # Prefer true numerics; _num on non-numeric str yields 0 — filter by type first
        nums = [
            float(x)
            for x in obj
            if (isinstance(x, (int, float)) and not isinstance(x, bool))
            or (isinstance(x, str) and _num(x, default=1e309) != 1e309)
        ]
        if nums:
            out[prefix + "__mean"] = float(np.mean(nums))
            out[prefix + "__max"] = float(np.max(nums))
            out[prefix + "__min"] = float(np.min(nums))
        for i, v in enumerate(obj[:12]):
            flatten_dict(v, f"{prefix}{i}.", out, depth + 1, also_leaf=also_leaf)
    elif isinstance(obj, bool):
        key = prefix[:-1]
        out[key] = float(int(obj))
        if also_leaf and key:
            _emit_aliases(out, key, float(int(obj)))
    elif isinstance(obj, (int, float)):
        key = prefix[:-1]
        out[key] = float(obj)
        out[key + "__log1p"] = float(np.log1p(abs(float(obj))))
        if also_leaf and key:
            _emit_aliases(out, key, float(obj))
            out["leaf:" + key.split(".")[-1] + "__log1p"] = float(np.log1p(abs(float(obj))))
    elif isinstance(obj, str):
        key = prefix[:-1]
        out[key + "=" + obj[:80]] = 1.0
        out[key + "__strlen"] = float(len(obj))
        n = _num(obj, default=1e309)
        if n != 1e309:
            out[key + "__num"] = float(n)
            # Also expose coerced numeric under the bare path so sealed heads that
            # trained on float leaves still see a value when presentation used strings.
            out[key] = float(n)
            out[key + "__log1p"] = float(np.log1p(abs(float(n))))
        if also_leaf and key:
            leaf = key.split(".")[-1]
            out["leaf:" + leaf + "=" + obj[:80]] = 1.0
            out["leaf:" + leaf + "__strlen"] = float(len(obj))
            if n != 1e309:
                out["leaf:" + leaf] = float(n)
            # last-two-segment alias e.g. invoice.total_usd
            parts = key.split(".")
            if len(parts) >= 2:
                short = ".".join(parts[-2:])
                out[short + "=" + obj[:80]] = 1.0
                out[short + "__strlen"] = float(len(obj))
                if n != 1e309:
                    out[short] = float(n)
                    out[short + "__log1p"] = float(np.log1p(abs(float(n))))
    elif obj is None:
        out[prefix + "__null"] = 1.0
    return out


def _emit_aliases(out: Dict[str, float], key: str, value: float) -> None:
    leaf = key.split(".")[-1]
    out["leaf:" + leaf] = value
    parts = key.split(".")
    if len(parts) >= 2:
        short = ".".join(parts[-2:])
        out[short] = value
        out[short + "__log1p"] = float(np.log1p(abs(value)))


def state_feature_dict(state: Any, workflow: str, qid: str, qtype: str) -> Dict[str, float]:
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {"_raw": state}
    if not isinstance(state, dict):
        state = {"_raw": str(state)}
    # Unwrap wrappers, canonicalize keys (incl. broader aliases), rebuild roots.
    state = prepare_state(state)

    d = flatten_dict(state)
    d["wf=" + workflow] = 1.0
    d["qid=" + qid] = 1.0
    d["qtype=" + qtype] = 1.0
    blob = json.dumps(state).lower()
    d["has_leave_lang"] = float(bool(LEAVE_RE.search(blob)))
    d["has_urgent_lang"] = float(bool(URGENT_RE.search(blob)))
    d["has_angry_lang"] = float(bool(ANGRY_RE.search(blob)))
    d["has_benign_lang"] = float(bool(BENIGN_RE.search(blob)))
    d["money_mentions"] = float(len(MONEY_RE.findall(blob)))
    d["blob_len"] = float(len(blob))

    if workflow == "invoice_processing":
        inv = dig(state, "invoice") or {}
        po = dig(state, "purchase_order") or {}
        if not isinstance(inv, dict):
            inv = {}
        if not isinstance(po, dict):
            po = {}
        inv_total = _num(inv.get("total_usd"))
        po_total = _num(po.get("total_usd"))
        d["var_usd"] = inv_total - po_total
        d["var_pct"] = ((inv_total - po_total) / po_total * 100.0) if po_total else 0.0
        # Ensure sealed path keys exist even if flatten prefixing differed.
        if inv_total:
            d["invoice.total_usd"] = inv_total
            d["invoice.total_usd__log1p"] = float(np.log1p(abs(inv_total)))
        if po_total:
            d["purchase_order.total_usd"] = po_total
            d["purchase_order.total_usd__log1p"] = float(np.log1p(abs(po_total)))
        pay = dig(state, "payment") or {}
        if not isinstance(pay, dict):
            pay = {}
        d["overdue"] = float(pay.get("status") == "overdue" or _num(pay.get("days_until_due")) < 0)
        d["days_until_due"] = _num(pay.get("days_until_due"))
        d["early_disc"] = float(bool(pay.get("early_payment_discount_pct")))
        inv_lines = inv.get("lines") or []
        po_lines = po.get("lines") or []
        if inv_lines and po_lines and isinstance(inv_lines[0], dict) and isinstance(po_lines[0], dict):
            d["qty_var"] = _num(inv_lines[0].get("qty")) - _num(po_lines[0].get("qty"))
            d["unit_var"] = _num(inv_lines[0].get("unit_usd")) - _num(po_lines[0].get("unit_usd"))
        deliv = dig(state, "delivery") or {}
        if not isinstance(deliv, dict):
            deliv = {}
        d["delivered"] = float(bool(deliv))
        d["recv_qty"] = _num(deliv.get("received_qty"))
        vh = dig(state, "vendor_history") or {}
        if not isinstance(vh, dict):
            vh = {}
        d["vendor_disputes"] = _num(vh.get("disputes_12m"))
        prior = vh.get("prior_invoice_ids") or []
        inv_id = inv.get("id")
        d["duplicate_id"] = float(inv_id in prior) if inv_id else 0.0

    if workflow == "agent_trace_observability":
        ts = dig(state, "trace_summary") or {}
        if not isinstance(ts, dict):
            ts = {}
        d["tool_errors"] = _num(ts.get("tool_errors"))
        d["violations"] = _num(ts.get("constraint_violations"))
        d["irreversible"] = _num(ts.get("irreversible_actions"))
        d["steps"] = _num(ts.get("steps"))
        d["duration"] = _num(ts.get("duration_s"))
        ag = dig(state, "agent") or {}
        if not isinstance(ag, dict):
            ag = {}
        d["autonomy=" + str(ag.get("autonomy"))] = 1.0
        cons = dig(state, "constraints")
        d["n_constraints"] = float(len(cons or []))

    if workflow == "security_incidents":
        pr = dig(state, "principal") or {}
        if not isinstance(pr, dict):
            pr = {}
        d["mfa"] = float(bool(pr.get("mfa_enrolled")))
        d["principal_type=" + str(pr.get("type"))] = 1.0
        priv = pr.get("privileges") or []
        d["n_priv"] = float(len(priv) if isinstance(priv, list) else 0)
        d["privileged"] = float(
            isinstance(priv, list)
            and any("admin" in str(p).lower() or "write" in str(p).lower() for p in priv)
        )
        hist = dig(state, "history") or {}
        if not isinstance(hist, dict):
            hist = {}
        d["prior_alerts"] = _num(hist.get("prior_alerts_90d"))
        d["rot_days"] = _num(hist.get("credential_rotation_days_ago"))
        d["logins_30d"] = _num(hist.get("logins_30d"))
        d["countries"] = _num(hist.get("distinct_countries_30d"))
        ctx = dig(state, "context") or {}
        if not isinstance(ctx, dict):
            ctx = {}
        d["allowlist"] = float(bool(ctx.get("source_on_allowlist")))
        d["change_window"] = float(bool(ctx.get("change_window_active")))
        d["crit=" + str(ctx.get("asset_criticality"))] = 1.0
        alert = dig(state, "alert") or {}
        if not isinstance(alert, dict):
            alert = {}
        d["rule=" + str(alert.get("rule"))] = 1.0
        evid = str(alert.get("evidence") or "") + " " + str(alert.get("description") or "")
        d["benign_evidence"] = float(bool(BENIGN_RE.search(evid)))

    if workflow == "customer_service":
        acc = dig(state, "account") or {}
        if not isinstance(acc, dict):
            acc = {}
        d["tier=" + str(acc.get("tier"))] = 1.0
        d["ltv"] = _num(acc.get("lifetime_value_usd"))
        d["prior_tix"] = _num(acc.get("prior_tickets_90d"))
        d["tenure"] = _num(acc.get("tenure_months"))
        d["seats"] = _num(acc.get("seats"))
        th = dig(state, "thread") or []
        if not isinstance(th, list):
            th = []
        d["turns"] = float(len(th))
        cust = " ".join(
            t.get("text", "") for t in th if isinstance(t, dict) and t.get("role") == "customer"
        )
        agent = " ".join(
            t.get("text", "") for t in th if isinstance(t, dict) and t.get("role") == "agent"
        )
        d["cust_len"] = float(len(cust))
        d["cust_leave"] = float(bool(LEAVE_RE.search(cust)))
        d["cust_angry"] = float(bool(ANGRY_RE.search(cust)))
        d["cust_urgent"] = float(bool(URGENT_RE.search(cust)))
        d["agent_len"] = float(len(agent))
        for topic, pat in [
            ("billing", r"\b(bill|invoice|charge|refund|payment)\b"),
            ("cancel_account", r"\b(cancel|close.*(account|subscription))\b"),
            ("technical", r"\b(bug|error|crash|not working|outage)\b"),
            ("security", r"\b(hack|breach|password|compromised|login)\b"),
        ]:
            d["topic_" + topic] = float(bool(re.search(pat, cust, re.I)))
    return d
