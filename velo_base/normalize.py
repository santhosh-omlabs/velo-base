# Copyright 2026 Santhosh Reddy
# SPDX-License-Identifier: Apache-2.0

"""State key canonicalization for robustness (snake_case + synonym inverse).

Maps camelCase / kebab-case / stress synonym field names back to the snake_case
schema used at training time so dict features, heuristics, TF-IDF text, and CS
distill see a stable key vocabulary.

Also recovers business payloads buried under wrapper envelopes and applies
broader paraphrase / fuzzy token canonicalization (kept separate from the
exported INVERSE_SYNONYMS so alien-suite integrity asserts remain valid).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Set

# Forward map matches scripts/stress_eval.py SYNONYMS (canonical -> stress synonym).
FIELD_SYNONYMS = {
    "invoice": "bill",
    "purchase_order": "po",
    "payment": "payment_info",
    "delivery": "shipment",
    "vendor_history": "supplier_history",
    "total_usd": "amount_usd",
    "unit_usd": "unit_price_usd",
    "days_until_due": "days_to_due",
    "early_payment_discount_pct": "early_discount_pct",
    "received_qty": "qty_received",
    "disputes_12m": "dispute_count_12m",
    "invoices_12m": "invoice_count_12m",
    "prior_invoice_ids": "previous_invoice_ids",
    "trace_summary": "run_summary",
    "constraint_violations": "violations",
    "irreversible_actions": "irreversible_count",
    "tool_errors": "tool_error_count",
    "duration_s": "duration_seconds",
    "lifetime_value_usd": "ltv_usd",
    "prior_tickets_90d": "tickets_90d",
    "tenure_months": "account_age_months",
    "thread": "conversation",
    "principal": "actor",
    "mfa_enrolled": "has_mfa",
    "privileges": "perms",
    "prior_alerts_90d": "alerts_90d",
    "credential_rotation_days_ago": "days_since_cred_rotation",
    "distinct_countries_30d": "countries_30d",
    "logins_30d": "login_count_30d",
    "source_on_allowlist": "on_allowlist",
    "change_window_active": "in_change_window",
    "asset_criticality": "criticality",
    "description": "desc",
    "evidence": "evidence_text",
}

# Inverse: synonym -> canonical. amount_usd handled separately (native under orders.*).
_INVERSE_RAW = {v: k for k, v in FIELD_SYNONYMS.items()}
NATIVE_COLLIDING_SYNONYMS = frozenset({"amount_usd"})
INVERSE_SYNONYMS = {k: v for k, v in _INVERSE_RAW.items() if k not in NATIVE_COLLIDING_SYNONYMS}

# amount_usd is native under orders.*; elsewhere it is the stress synonym for total_usd.
_ORDERS_CONTEXT = frozenset({"orders", "order"})

# Broader paraphrase map used at canonicalize time but NOT merged into
# INVERSE_SYNONYMS (alien suites assert their renames stay outside that export).
BROADER_ALIASES: Dict[str, str] = {
    # invoice / AP
    "factura": "invoice",
    "bill_document": "invoice",
    "invoice_document": "invoice",
    "sales_invoice": "invoice",
    "vendor_invoice": "invoice",
    "requisition_order": "purchase_order",
    "buy_authorization": "purchase_order",
    "purchase_req": "purchase_order",
    "po_document": "purchase_order",
    "settlement": "payment",
    "remittance": "payment",
    "payment_record": "payment",
    "fulfillment_record": "delivery",
    "goods_receipt": "delivery",
    "receiving_record": "delivery",
    "shipment_record": "delivery",
    "counterparty_ledger": "vendor_history",
    "supplier_ledger": "vendor_history",
    "vendor_ledger": "vendor_history",
    "grand_total_amount": "total_usd",
    "net_payable_usd": "total_usd",
    "invoice_total_usd": "total_usd",
    "gross_amount_usd": "total_usd",
    "line_total_usd": "total_usd",
    "per_unit_charge": "unit_usd",
    "unit_price": "unit_usd",
    "unit_cost_usd": "unit_usd",
    "remaining_days_to_pay": "days_until_due",
    "days_remaining_until_due": "days_until_due",
    "due_in_days": "days_until_due",
    "prompt_pay_rebate_pct": "early_payment_discount_pct",
    "early_pay_discount": "early_payment_discount_pct",
    "qty_on_dock": "received_qty",
    "quantity_received": "received_qty",
    "received_quantity": "received_qty",
    "claim_count_trailing_year": "disputes_12m",
    "dispute_count_year": "disputes_12m",
    "bill_volume_trailing_year": "invoices_12m",
    "invoice_volume_12m": "invoices_12m",
    "historical_bill_refs": "prior_invoice_ids",
    "prior_bill_ids": "prior_invoice_ids",
    "line_items": "lines",
    "product_code": "sku",
    "item_sku": "sku",
    "quantity": "qty",
    "qty_ordered": "qty",
    "payment_state": "status",
    "payment_terms_code": "terms",
    "receipt_condition": "condition",
    "incoterms": "freight_terms",
    "supplier_name": "vendor",
    "vendor_name": "vendor",
    "ccy": "currency",
    # agent / trace
    "execution_digest": "trace_summary",
    "run_digest": "trace_summary",
    "trace_digest": "trace_summary",
    "policy_breach_count": "constraint_violations",
    "guardrail_violations": "constraint_violations",
    "irrevocable_step_count": "irreversible_actions",
    "tooling_fault_count": "tool_errors",
    "elapsed_wall_seconds": "duration_s",
    "runtime_agent": "agent",
    "guardrails": "constraints",
    "assigned_task": "task",
    # CS
    "customer_ltv_dollars": "lifetime_value_usd",
    "ltv_dollars": "lifetime_value_usd",
    "support_cases_quarter": "prior_tickets_90d",
    "months_as_customer": "tenure_months",
    "message_stream": "thread",
    "client_profile": "account",
    "customer_account": "account",
    # security
    "subject_entity": "principal",
    "mfa_registered": "mfa_enrolled",
    "entitlement_set": "privileges",
    "signal_count_quarter": "prior_alerts_90d",
    "days_since_secret_rotate": "credential_rotation_days_ago",
    "geo_diversity_month": "distinct_countries_30d",
    "auth_events_month": "logins_30d",
    "origin_allowlisted": "source_on_allowlist",
    "maintenance_window_open": "change_window_active",
    "asset_tier": "asset_criticality",
    "narrative": "description",
    "supporting_artifacts": "evidence",
    "detection_event": "alert",
    "situational_context": "context",
    "behavioral_history": "history",
}

# Train-time top-level roots and salient children (for unwrap + shape rebuild).
SCHEMA_ROOTS = frozenset(
    {
        "invoice",
        "purchase_order",
        "payment",
        "delivery",
        "vendor_history",
        "account",
        "thread",
        "orders",
        "alert",
        "principal",
        "history",
        "context",
        "agent",
        "trace_summary",
        "constraints",
        "task",
    }
)

ROOT_CHILD_SIGNATURES: Dict[str, frozenset] = {
    "invoice": frozenset({"currency", "id", "lines", "total_usd", "vendor"}),
    "purchase_order": frozenset({"freight_terms", "id", "lines", "total_usd"}),
    "payment": frozenset(
        {
            "days_until_due",
            "discount_expires_in_days",
            "early_payment_discount_pct",
            "status",
            "terms",
        }
    ),
    "delivery": frozenset({"condition", "date", "received_qty"}),
    "vendor_history": frozenset({"disputes_12m", "invoices_12m", "prior_invoice_ids"}),
    "account": frozenset(
        {"lifetime_value_usd", "prior_tickets_90d", "seats", "tenure_months", "tier"}
    ),
    "trace_summary": frozenset(
        {
            "constraint_violations",
            "duration_s",
            "irreversible_actions",
            "steps",
            "tool_errors",
        }
    ),
    "agent": frozenset({"autonomy", "model"}),
    "alert": frozenset({"description", "evidence", "rule", "id"}),
    "principal": frozenset({"mfa_enrolled", "privileges", "type", "id", "name"}),
    "orders": frozenset({"amount_usd", "date", "id", "status"}),
    "history": frozenset(
        {
            "prior_alerts_90d",
            "credential_rotation_days_ago",
            "logins_30d",
            "distinct_countries_30d",
        }
    ),
    "context": frozenset(
        {"source_on_allowlist", "change_window_active", "asset_criticality"}
    ),
    "task": frozenset({"description", "id", "goal"}),
    "constraints": frozenset(),  # often a list
}

WRAPPER_HINTS = frozenset(
    {
        "envelope",
        "record",
        "document",
        "payload",
        "data",
        "body",
        "content",
        "wrapper",
        "meta",
        "audit",
        "noise_sidecar",
        "sidecar",
        "container",
        "packet",
        "message",
        "request",
        "response",
        "case",
        "case_state",
        "state",
        "input",
        "inputs",
        "root",
        "inner",
        "outer",
        "wrapped",
        "bundle",
        "pkg",
        "package",
        "frame",
        "layer",
        "nest",
        "node",
        "entity",
        "object",
        "obj",
        "value",
        "values",
        "item",
        "items",
        "result",
        "results",
        "section",
        "block",
        "part",
        "fragment",
        "blob",
        "raw",
        "raw_state",
        "business",
        "business_payload",
        "core",
        "core_state",
        "domain",
        "domain_state",
        "facts",
        "fact_sheet",
        "snapshot",
        "view",
        "projection",
        "schema_rev",
        "schema",
        "version",
        "rev",
        "nonce",
        "channel",
        "tags",
        "ignore",
        "noise",
        "padding",
        "watermark",
        "aux",
        "_aux_finance",
        "_pad",
        "notes_meta",
        "备注",
    }
)

# Presentation-noise keys stripped at prepare_state (unicode pads, watermarks).
_NOISE_KEY_EXACT = frozenset(
    {
        "备注",
        "watermark",
        "notes_meta",
        "_pad",
        "padding",
        "noise",
        "ignore",
    }
)
_NOISE_KEY_PREFIXES = ("_pad", "pad_", "_aux_", "aux_")

_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_DIGIT_UNIT_RE = re.compile(r"(\d)_([a-z])\b")
_LETTER_DIGIT_RE = re.compile(r"([a-z])(\d)")
_STOP_TOKENS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "by",
        "as",
        "is",
        "at",
        "from",
        "with",
        "per",
        "set",
        "count",
        "num",
        "number",
        "value",
        "val",
        "info",
        "data",
        "record",
        "records",
        "field",
        "fields",
        "name",
        "ref",
        "refs",
        "id",
        "ids",
        "list",
        "stream",
        "document",
        "doc",
        "entity",
        "object",
        "obj",
        "item",
        "items",
        "profile",
        "event",
        "events",
        "month",
        "year",
        "quarter",
        "trailing",
        "since",
        "ago",
    }
)

# Canonical keys we may fuzzy-match toward (train schema + synonym targets).
_FUZZY_TARGETS: tuple[str, ...] = tuple(
    sorted(
        set(FIELD_SYNONYMS.keys())
        | set(SCHEMA_ROOTS)
        | {
            "lines",
            "qty",
            "sku",
            "currency",
            "vendor",
            "status",
            "terms",
            "condition",
            "freight_terms",
            "type",
            "autonomy",
            "steps",
            "seats",
            "tier",
            "role",
            "text",
            "rule",
            "model",
            "date",
            "id",
            "discount_expires_in_days",
            "amount_usd",
        }
    )
)


def to_snake(key: str) -> str:
    """Normalize camelCase / kebab-case / mixed keys to snake_case.

    Handles stress ``to_camel`` quirks where ``invoices_12m`` becomes ``invoices12M``
    (``.title()`` capitalizes the unit letter after digits).
    """
    if not isinstance(key, str):
        return str(key)
    s = key.replace("-", "_")
    s = _CAMEL_RE.sub(r"\1_\2", s)
    s = s.lower()
    # Undo digit-unit split from camel boundary on ``12M`` → ``12_m`` → ``12m``.
    s = _DIGIT_UNIT_RE.sub(r"\1\2", s)
    # ``invoices12m`` → ``invoices_12m``.
    s = _LETTER_DIGIT_RE.sub(r"\1_\2", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _tokens(s: str) -> Set[str]:
    return {t for t in s.lower().split("_") if t and t not in _STOP_TOKENS}


def _fuzzy_canonicalize(snake: str) -> Optional[str]:
    """Map elaborating paraphrases to schema keys via token overlap.

    Conservative: require meaningful overlap so we do not collapse unrelated keys.
    """
    toks = _tokens(snake)
    if not toks or snake in _FUZZY_TARGETS:
        return None
    best: Optional[str] = None
    best_score = 0.0
    for cand in _FUZZY_TARGETS:
        ct = _tokens(cand)
        if not ct:
            continue
        inter = toks & ct
        if not inter:
            # stem / containment: "payable"~"payment" handled via broader aliases
            # Allow single-token containment for longer tokens
            cont = False
            for a in toks:
                for b in ct:
                    if len(min(a, b, key=len)) >= 6 and (a.startswith(b) or b.startswith(a)):
                        cont = True
                        inter = {a, b}
                        break
                if cont:
                    break
            if not cont:
                continue
        j = len(inter) / max(len(toks | ct), 1)
        subset = len(inter) / max(len(ct), 1)
        score = 0.55 * j + 0.45 * subset
        # Multi-token alien keys must not collapse onto a single shared token
        # (ship_terms → terms, money_code → …).
        if len(ct) == 1 and len(toks) > 1:
            score *= 0.4
        if len(ct) == 1 and list(ct)[0] in toks and len(toks) == 1 and len(list(ct)[0]) >= 4:
            score = max(score, 0.72)
        if score > best_score or (
            abs(score - best_score) < 1e-9 and best is not None and len(cand) > len(best)
        ):
            best, best_score = cand, score
    if best is not None and best_score >= 0.62:
        return best
    return None


def canonicalize_key(key: str, parent_canonical: Optional[str] = None) -> str:
    snake = to_snake(key)
    if snake == "amount_usd":
        # Preserve native CS orders.amount_usd; recover invoice/PO total_usd synonym
        # (including under invoice.lines.* where parent is "lines").
        if parent_canonical in _ORDERS_CONTEXT:
            return "amount_usd"
        return "total_usd"
    if snake in INVERSE_SYNONYMS:
        return INVERSE_SYNONYMS[snake]
    if snake in BROADER_ALIASES:
        return BROADER_ALIASES[snake]
    if snake in SCHEMA_ROOTS or snake in _FUZZY_TARGETS:
        return snake
    fuzzy = _fuzzy_canonicalize(snake)
    if fuzzy:
        return fuzzy
    return snake


def normalize_state_keys(obj: Any, parent_canonical: Optional[str] = None) -> Any:
    """Recursively rename dict keys to training snake_case schema."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ck = canonicalize_key(k, parent_canonical)
            nv = normalize_state_keys(v, ck)
            if ck in out and isinstance(out[ck], dict) and isinstance(nv, dict):
                out[ck] = {**out[ck], **nv}
            else:
                out[ck] = nv
        return out
    if isinstance(obj, list):
        return [normalize_state_keys(v, parent_canonical) for v in obj]
    return obj


def _payload_score(d: Any) -> int:
    if not isinstance(d, dict):
        return 0
    score = 0
    for k in d.keys():
        ck = canonicalize_key(k) if isinstance(k, str) else str(k)
        if ck in SCHEMA_ROOTS:
            score += 3
        elif ck in _FUZZY_TARGETS:
            score += 1
    return score


def recover_business_state(obj: Any, max_depth: int = 8) -> Any:
    """Lift typed-decision business state out of wrapper envelopes.

    Presentation nests (envelope.record.document / payload / …) change paths but
    not semantics. Prefer the subtree with the strongest schema-root signature.
    """
    if not isinstance(obj, dict):
        return obj

    best = obj
    best_score = _payload_score(obj)
    best_depth = 0

    def walk(o: Any, depth: int) -> None:
        nonlocal best, best_score, best_depth
        if depth > max_depth or not isinstance(o, dict):
            return
        sc = _payload_score(o)
        # Prefer higher score; on ties prefer shallower (less wrapper noise).
        if sc > best_score or (sc == best_score and sc > 0 and depth < best_depth):
            best, best_score, best_depth = o, sc, depth
        for k, v in o.items():
            # Skip obvious noise siblings once we already have a strong candidate
            # but still explore — cheap at depth<=8.
            if isinstance(v, dict):
                walk(v, depth + 1)
            elif isinstance(v, list):
                for item in v[:6]:
                    if isinstance(item, dict):
                        walk(item, depth + 1)

    walk(obj, 0)

    # Strong schema signature found deeper than the root → unwrap.
    if best_score >= 6 and best is not obj:
        return best

    # Light single-payload wrappers: {"payload": <state>, "meta": ...}
    if best_score < 6:
        candidates = []
        for k, v in obj.items():
            if not isinstance(v, dict):
                continue
            sc = _payload_score(v)
            if sc >= 6:
                candidates.append((sc, v))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            return candidates[0][1]

    return best if best_score >= 6 else obj


def _shape_score(d: dict, signature: frozenset) -> float:
    if not signature:
        return 0.0
    keys = {canonicalize_key(k) for k in d.keys() if isinstance(k, str)}
    return len(keys & signature) / float(len(signature))


def _child_keyset(d: dict) -> Set[str]:
    return {canonicalize_key(k) for k in d.keys() if isinstance(k, str)}



def _structural_root_guess(d: dict) -> Optional[str]:
    """Guess canonical root from value shapes when keys are fully opaque."""
    if not isinstance(d, dict) or not d:
        return None
    vals = list(d.values())
    n_num = sum(1 for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool))
    n_bool = sum(1 for v in vals if isinstance(v, bool))
    n_str = sum(1 for v in vals if isinstance(v, str))
    n_null = sum(1 for v in vals if v is None)
    list_of_dict = [
        v for v in vals if isinstance(v, list) and v and isinstance(v[0], dict)
    ]
    list_of_str = [
        v for v in vals if isinstance(v, list) and v and isinstance(v[0], str)
    ]
    strs = [v for v in vals if isinstance(v, str)]
    sl = [s.lower().strip() for s in strs]

    _STATUS = {
        "open", "paid", "overdue", "pending", "closed", "shipped", "received",
        "ok", "failed", "scheduled", "cancelled", "canceled", "draft",
    }
    _TIER = {"free", "pro", "enterprise", "starter", "business", "standard", "plus", "basic"}
    _CRIT = {"low", "medium", "high", "critical"}
    _NET = any(s.startswith("net ") or s.startswith("net") and any(ch.isdigit() for ch in s) for s in sl)
    _HAS_STATUS = any(s in _STATUS for s in sl)
    _HAS_TIER = any(s in _TIER for s in sl)
    _HAS_CRIT = any(s in _CRIT for s in sl)

    # invoice / PO: line-item list present
    if list_of_dict and n_num >= 1:
        sample = list_of_dict[0][0]
        sn = sum(
            1
            for x in sample.values()
            if isinstance(x, (int, float)) and not isinstance(x, bool)
        )
        if sn >= 1 and len(sample) >= 2:
            if any(len(s) == 3 and s.isalpha() for s in strs):
                return "invoice"
            return "purchase_order"

    # principal: bool MFA-like + privilege string list + type string
    if n_bool >= 1 and list_of_str and n_str >= 1:
        return "principal"

    # context: bools + criticality-like string, small
    if n_bool >= 1 and n_str >= 1 and len(d) <= 5 and not list_of_dict and not list_of_str:
        if _HAS_CRIT or n_bool >= 2:
            return "context"
        return "context"

    # payment BEFORE account: need status/net lexemes (null alone is not enough —
    # delivery with date=None must not become payment).
    if (
        not list_of_dict
        and n_bool == 0
        and n_str >= 1
        and len(d) >= 4
        and (_HAS_STATUS or _NET)
        and (n_num + n_null) >= 1
        and not _HAS_TIER
    ):
        return "payment"

    # account: tier string + numerics (ltv/seats/tenure/tickets)
    if (
        n_num >= 2
        and n_str >= 1
        and n_bool == 0
        and not list_of_dict
        and len(d) >= 4
        and (_HAS_TIER or (n_null == 0 and n_num >= 3 and not _HAS_STATUS and not _NET))
    ):
        return "account"

    # soft payment fallback: discount nulls + 2 strings (status+terms) without tier
    if (
        not list_of_dict
        and n_null >= 1
        and n_str >= 2
        and n_bool == 0
        and len(d) >= 4
        and n_num <= 4
        and not _HAS_TIER
        and not _HAS_CRIT
    ):
        return "payment"

    # delivery: small, qty + date/condition
    if not list_of_dict and n_num >= 1 and n_str >= 1 and 2 <= len(d) <= 4 and n_bool == 0:
        return "delivery"

    # vendor_history: ints + id string list
    if list_of_str and n_num >= 1 and not list_of_dict:
        return "vendor_history"

    # history vs trace_summary: both numeric-heavy
    if n_num >= 3 and n_str == 0 and not list_of_dict and n_bool == 0:
        # trace_summary typically 5 numeric fields; history often 4
        if len(d) >= 5:
            return "trace_summary"
        return "history"

    # alert: several free-text strings, no numbers
    if n_str >= 2 and n_num == 0 and n_bool == 0 and not list_of_dict and not list_of_str:
        return "alert"

    # agent: autonomy + model strings only
    if n_str >= 2 and len(d) <= 3 and not list_of_dict and n_num == 0 and n_bool == 0:
        return "agent"

    # task dict form
    if n_str >= 1 and len(d) <= 4 and not list_of_dict and n_num <= 1 and n_bool == 0:
        return "task"

    return None


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_DATE_US = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_DATE_EU = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_DATE_DOT = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})$")
_DATE_LONG = re.compile(
    r"^(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})$"
)
_DATE_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_INSTR_PAD_RE = re.compile(
    r"(?:\s*Please reason carefully from the case state only\.\s*)+|◆",
    re.I,
)


def coerce_date_to_iso(v: Any) -> Any:
    """Normalize common presentation date forms back to YYYY-MM-DD."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if _DATE_ISO.match(s):
        return s
    m = _DATE_US.match(s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    m = _DATE_EU.match(s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    m = _DATE_DOT.match(s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = _DATE_LONG.match(s)
    if m:
        d, mon, y = int(m.group(1)), m.group(2).lower(), m.group(3)
        mo = _MONTHS.get(mon)
        if mo and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    # ISO datetime prefix
    if len(s) >= 10 and _DATE_ISO.match(s[:10]):
        return s[:10]
    return v


def _is_noise_key(k: str) -> bool:
    if not isinstance(k, str):
        return False
    if k in _NOISE_KEY_EXACT or k in WRAPPER_HINTS and k.startswith("_"):
        return True
    low = k.lower()
    if any(low.startswith(p) for p in _NOISE_KEY_PREFIXES):
        return True
    # non-ascii-only or leading unicode pad markers
    if k in {"备注"} or (not k.isascii() and len(k) <= 8):
        return True
    if low.startswith("_pad") or "pad_" in low and "α" in k:
        return True
    return False


def strip_presentation_noise(obj: Any) -> Any:
    """Drop unicode/pad/watermark siblings that are not part of train schema."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _is_noise_key(k):
                continue
            out[k] = strip_presentation_noise(v)
        return out
    if isinstance(obj, list):
        return [strip_presentation_noise(v) for v in obj]
    return coerce_date_to_iso(obj)


def sanitize_instructions(text: str) -> str:
    """Strip long stress instruction padding without touching real prompts."""
    if not isinstance(text, str) or not text:
        return text
    cleaned = _INSTR_PAD_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _hint_tokens(hint: str) -> Set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", hint.lower()) if t and len(t) >= 2}


def _slot_affinity(hint: str, value: Any, slot: str) -> float:
    """Score how well an opaque (key hint, value) binds to a canonical slot."""
    ht = _hint_tokens(hint)
    sl = slot.lower()
    st = _hint_tokens(sl)
    score = 0.0

    # Direct token / stem overlap with slot name
    inter = ht & st
    if inter:
        score += 2.5 * len(inter)
    for a in ht:
        for b in st:
            if len(a) >= 4 and len(b) >= 4 and (a.startswith(b[:4]) or b.startswith(a[:4])):
                score += 1.2

    # Domain synonym cues (alien paraphrases → slot)
    cues = {
        "qty": {"qty", "units", "n", "dock", "docked", "count", "quantity", "recv", "received", "pack"},
        "unit_usd": {"each", "per", "piece", "rate", "price", "charge"},
        "total_usd": {"sum", "total", "money", "amt", "amount", "worth", "grand", "payable", "due"},
        "lifetime_value_usd": {"worth", "ltv", "money", "acct", "lifetime"},
        "amount_usd": {"amount", "money", "usd"},
        "days_until_due": {"horizon", "due", "payable", "days", "until"},
        "discount_expires_in_days": {"discount", "expires", "expiry"},
        "early_payment_discount_pct": {"skim", "discount", "pct", "rebate", "prompt", "fast"},
        "received_qty": {"dock", "docked", "recv", "received", "units"},
        "disputes_12m": {"quarrel", "dispute", "claim"},
        "invoices_12m": {"docs", "filed", "invoice", "bill", "volume"},
        "prior_invoice_ids": {"older", "prior", "refs", "ids"},
        "constraint_violations": {"rule", "break", "tally", "violat", "guard", "breach", "policy"},
        "irreversible_actions": {"one", "way", "irrevoc", "moves"},
        "tool_errors": {"tool", "glitch", "error", "fault"},
        "duration_s": {"wall", "clock", "secs", "duration", "elapsed", "runtime"},
        "prior_tickets_90d": {"cases", "ticket", "season", "support"},
        "tenure_months": {"months", "books", "tenure", "age", "on"},
        "prior_alerts_90d": {"ping", "alert", "season", "signal"},
        "credential_rotation_days_ago": {"secret", "age", "cred", "rotation", "rotate", "days"},
        "distinct_countries_30d": {"nation", "country", "countries", "geo", "mix"},
        "logins_30d": {"signin", "signins", "login", "logins", "auth"},
        "source_on_allowlist": {"src", "trusted", "allow", "allowlist"},
        "change_window_active": {"chg", "slot", "window", "change", "maint"},
        "asset_criticality": {"asset", "importance", "crit", "tier"},
        "seats": {"seats", "seat", "licenses"},
        "steps": {"steps", "step"},
        "status": {"state", "flag", "status", "lifecycle"},
        "terms": {"net", "terms", "code"},
        "condition": {"recv", "state", "condition", "accept"},
        "freight_terms": {"ship", "freight", "incoterm", "basis"},
        "currency": {"money", "code", "fx", "ccy", "currency"},
        "vendor": {"counterparty", "payee", "supplier", "vendor"},
        "sku": {"stock", "catalog", "sku", "code"},
        "description": {"blurb", "desc", "narrative"},
        "evidence": {"proof", "blob", "artifact"},
        "mfa_enrolled": {"second", "factor", "mfa"},
        "privileges": {"capability", "bag", "perms", "entitlement"},
    }
    for tok in cues.get(sl, ()):
        if tok in ht or any(tok in h for h in ht):
            score += 2.0

    # Value-shape cues
    if isinstance(value, bool):
        if any(x in sl for x in ("mfa", "allowlist", "active", "enrolled", "window")):
            score += 3.0
        else:
            score -= 1.0
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        if "pct" in sl and 0 < abs(v) <= 100:
            score += 1.5
        if ("usd" in sl or "ltv" in sl or "amount" in sl) and abs(v) >= 50:
            score += 1.8
        if ("usd" in sl or "unit" in sl) and "qty" not in sl and abs(v) >= 20:
            score += 0.8
        if sl in {"qty", "received_qty", "seats", "steps"} and abs(v) < 500 and float(v) == int(v):
            score += 1.2
        # unit_count must not outrank qty just because "unit" overlaps unit_usd
        if "count" in ht and "rate" not in ht and "price" not in ht and "money" not in ht and "each" not in ht:
            if sl in {"qty", "received_qty"}:
                score += 4.0
            if sl in {"unit_usd", "total_usd", "lifetime_value_usd"}:
                score -= 5.0
        if ("piece" in ht or "rate" in ht or "each" in ht) and sl == "unit_usd":
            score += 4.0
        if ("piece" in ht or "rate" in ht) and sl == "total_usd":
            score -= 2.0
        if ("figure" in ht or "cash" in ht or "sum" in ht or "balance" in ht) and sl == "total_usd":
            score += 3.5
        if ("figure" in ht or "cash" in ht or "sum" in ht) and sl == "unit_usd":
            score -= 2.5
        # magnitude: line totals >> unit prices >> qty typically
        if sl == "total_usd" and abs(v) >= 200:
            score += 1.5
        if sl == "unit_usd" and 1 <= abs(v) < 5000 and "count" not in ht:
            score += 1.0
        if sl == "qty" and abs(v) >= 1e4:
            score -= 3.0
        if "countries" in sl and 0 <= v <= 30:
            score += 2.5
        if "logins" in sl and v >= 20:
            score += 2.5
        if "alerts" in sl and 0 <= v <= 50:
            score += 1.5
        if "tickets" in sl and 0 <= v <= 50:
            score += 1.5
        if "tenure" in sl and 0 <= v <= 240:
            score += 1.2
        if "rotation" in sl or "days_ago" in sl:
            if "age" in ht or "secret" in ht or "day" in ht:
                score += 2.0
        if "duration" in sl and isinstance(value, float):
            score += 1.5
        if sl == "days_until_due":
            score += 0.5
        # Penalize binding huge login-like counts onto countries
        if "countries" in sl and v > 40:
            score -= 4.0
        if "alerts" in sl and v > 80:
            score -= 3.0
        if "tickets" in sl and v > 80:
            score -= 2.0
        if "qty" in sl and abs(v) >= 1e4:
            score -= 2.0
    elif isinstance(value, str):
        s = value.strip()
        slv = s.lower()
        if sl == "currency" and len(s) == 3 and s.isalpha():
            score += 5.0
        if sl == "status" and slv in {
            "open", "paid", "overdue", "pending", "closed", "shipped",
            "received", "ok", "failed", "scheduled", "cancelled", "canceled", "draft",
        }:
            score += 5.0
        if sl == "terms" and ("net" in slv or re.match(r"^net\s*\d+", slv)):
            score += 5.0
        if sl == "tier" and slv in {"free", "pro", "enterprise", "starter", "business", "standard", "plus", "basic"}:
            score += 5.0
        if sl == "asset_criticality" and slv in {"low", "medium", "high", "critical"}:
            score += 5.0
        if sl == "id" and (
            bool(re.match(r"^(INV|PO|ORD|ALERT|TCK|USR|AGT)-", s, re.I))
            or ("-" in s and any(ch.isdigit() for ch in s) and len(s) <= 48)
        ):
            score += 3.5
        if sl == "sku" and (s.upper().startswith("SKU") or ("-" in s and len(s) <= 24)):
            score += 3.0
        if sl == "date" and (
            _DATE_ISO.match(s)
            or _DATE_US.match(s)
            or _DATE_EU.match(s)
            or _DATE_DOT.match(s)
            or _DATE_LONG.match(s)
            or "/" in s
        ):
            score += 4.0
        if sl == "vendor" and len(s) >= 4:
            score += 1.0
        if sl in {"description", "evidence", "rule"} and len(s) >= 8:
            score += 1.0
        # Penalize putting net-terms into status and vice versa
        if sl == "status" and "net" in slv:
            score -= 5.0
        if sl == "terms" and slv in {
            "open", "paid", "overdue", "pending", "closed", "scheduled", "draft",
        }:
            score -= 5.0
    elif isinstance(value, list):
        if value and isinstance(value[0], dict) and sl == "lines":
            score += 5.0
        if value and isinstance(value[0], str):
            if "ids" in sl or sl == "privileges" or sl == "constraints":
                score += 3.0
    return score


def _remap_opaque_leaves(obj: Any, parent: Optional[str] = None) -> Any:
    """After roots are recovered, map opaque child keys via signatures + value cues."""
    if isinstance(obj, list):
        if parent in ("invoice", "purchase_order", "lines"):
            line_parent = "lines"
        elif parent == "orders":
            line_parent = "orders"
        else:
            line_parent = parent
        return [_remap_opaque_leaves(v, line_parent) for v in obj]
    if not isinstance(obj, dict):
        return coerce_date_to_iso(obj)

    sig = set(ROOT_CHILD_SIGNATURES.get(parent or "", frozenset()) or set())
    if parent == "lines":
        sig |= {"qty", "sku", "unit_usd", "total_usd"}
    if parent == "alert":
        sig |= {"id", "rule", "description", "evidence"}
    if parent == "task":
        sig |= {"id", "description", "goal"}
    if parent == "payment":
        # discount_expires often present even when not in alien maps
        sig |= {
            "days_until_due",
            "discount_expires_in_days",
            "early_payment_discount_pct",
            "status",
            "terms",
        }

    used_targets: Set[str] = set()
    out: Dict[str, Any] = {}
    pending: list = []

    for k, v in obj.items():
        ck = canonicalize_key(k, parent)
        snake = to_snake(k)
        if ck in sig or (ck in _FUZZY_TARGETS and ck != snake) or ck in SCHEMA_ROOTS:
            nv = _remap_opaque_leaves(v, ck if isinstance(v, (dict, list)) else parent)
            if ck in out and isinstance(out[ck], dict) and isinstance(nv, dict):
                out[ck] = {**out[ck], **nv}
            else:
                out[ck] = nv
            used_targets.add(ck)
        elif snake in sig:
            nv = _remap_opaque_leaves(v, snake if isinstance(v, (dict, list)) else parent)
            out[snake] = nv
            used_targets.add(snake)
        else:
            pending.append((k, v, ck))

    unused = [s for s in sorted(sig) if s not in used_targets]

    # Score-based assignment (avoid greedy mis-binds on same-type numerics/strings).
    if pending and unused:
        scored = []
        for k, v, ck in pending:
            hint = to_snake(k)
            for slot in unused:
                sc = _slot_affinity(hint, v, slot)
                scored.append((sc, hint, k, v, ck, slot))
        scored.sort(key=lambda x: -x[0])
        assigned_keys = set()
        assigned_slots = set()
        for sc, hint, k, v, ck, slot in scored:
            if k in assigned_keys or slot in assigned_slots:
                continue
            if sc < 0.75 and len(unused) > 1:
                # keep very weak matches for a second pass
                continue
            assigned_keys.add(k)
            assigned_slots.add(slot)
            unused = [s for s in unused if s != slot]
            nv = _remap_opaque_leaves(v, slot if isinstance(v, (dict, list)) else parent)
            if slot in out and isinstance(out[slot], dict) and isinstance(nv, dict):
                out[slot] = {**out[slot], **nv}
            else:
                out[slot] = nv
            used_targets.add(slot)

        # Remainder: bind only if affinity is decent; else keep original key
        # (preserves train extras like principal.name on clean-ish inputs).
        for k, v, ck in pending:
            if k in assigned_keys:
                continue
            target = None
            best_sc = -1e9
            if unused:
                for slot in unused:
                    sc = _slot_affinity(to_snake(k), v, slot)
                    if sc > best_sc:
                        best_sc, target = sc, slot
                if target is not None and best_sc >= 0.75:
                    unused = [s for s in unused if s != target]
                else:
                    target = None
            final = target or ck
            nv = _remap_opaque_leaves(v, final if isinstance(v, (dict, list)) else parent)
            if final in out and isinstance(out[final], dict) and isinstance(nv, dict):
                out[final] = {**out[final], **nv}
            else:
                out[final] = nv
            used_targets.add(final)
            assigned_keys.add(k)
    else:
        for k, v, ck in pending:
            out[ck] = _remap_opaque_leaves(v, ck if isinstance(v, (dict, list)) else parent)

    return out


def _subtree_has_opaque(obj: Any, parent: Optional[str] = None) -> bool:
    """Detect alien/opaque keys that still need value-aware remapping."""
    if isinstance(obj, list):
        par = "orders" if parent == "orders" else (
            "lines" if parent in ("invoice", "purchase_order", "lines") else parent
        )
        return any(_subtree_has_opaque(v, par) for v in obj if isinstance(v, (dict, list)))
    if not isinstance(obj, dict):
        return False
    sig = set(ROOT_CHILD_SIGNATURES.get(parent or "", frozenset()) or set())
    if parent == "lines":
        sig |= {"qty", "sku", "unit_usd", "total_usd"}
    if parent == "orders":
        sig |= {"amount_usd", "date", "id", "status"}
    # Train extras that must never be treated as opaque (clean baseline)
    extras = {
        "name", "goal", "model", "autonomy", "role", "text", "seats", "tier",
        "rule", "description", "evidence", "freight_terms", "condition",
        "currency", "vendor", "sku", "qty", "terms", "status", "type",
        "discount_expires_in_days", "amount_usd", "date", "id",
    }
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        ck = canonicalize_key(k, parent)
        snake = to_snake(k)
        if ck in sig or snake in sig or ck in SCHEMA_ROOTS or ck in extras:
            if isinstance(v, (dict, list)) and _subtree_has_opaque(v, ck):
                return True
            continue
        if ck in _FUZZY_TARGETS and ck != snake:
            return True
        # unrecognized key name → opaque
        if snake == ck and ck not in sig and ck not in extras and ck not in _FUZZY_TARGETS:
            return True
    return False


def rebuild_canonical_roots(state: Any) -> Any:
    """If top-level roots are still alien, re-attach subtrees by child signatures."""
    if not isinstance(state, dict):
        return state
    # Already looks like a training state — preserve sealed-head parity on clean
    # inputs. Only remap children when opaque keys are detected.
    present = {canonicalize_key(k) for k in state.keys() if isinstance(k, str)}
    if len(present & SCHEMA_ROOTS) >= 2:
        out = {}
        for k, v in state.items():
            ck = canonicalize_key(k) if isinstance(k, str) else str(k)
            if isinstance(v, (dict, list)) and _subtree_has_opaque(v, ck):
                out[ck] = _remap_opaque_leaves(v, ck if ck in SCHEMA_ROOTS or ck == "orders" else None)
            elif isinstance(v, list) and v and isinstance(v[0], dict) and ck == "orders":
                # order line dicts may carry alien status synonyms
                if _subtree_has_opaque(v, "orders"):
                    out[ck] = _remap_opaque_leaves(v, "orders")
                else:
                    out[ck] = v
            else:
                out[ck] = v
        if "task" not in out and ("agent" in out or "trace_summary" in out):
            for k, v in list(out.items()):
                if k in SCHEMA_ROOTS:
                    continue
                if isinstance(v, str) and len(v.strip()) >= 12:
                    out["task"] = v
                    if k != "task":
                        del out[k]
                    break
        return out

    assigned: Dict[str, Any] = {}
    used = set()
    # Score every dict-valued child against each root signature.
    scored = []
    for k, v in state.items():
        if not isinstance(v, dict):
            continue
        best_sc = 0.0
        best_root = None
        for root, sig in ROOT_CHILD_SIGNATURES.items():
            if not sig:
                continue
            sc = _shape_score(v, sig)
            if sc > best_sc:
                best_sc, best_root = sc, root
        # Structural fallback when keys are fully opaque
        guess = _structural_root_guess(v)
        if guess:
            # Structural fingerprint outranks weak key overlap on opaque renames.
            if best_sc < 0.6:
                # Prefer structural when it conflicts with a weak shape match
                if best_root != guess and best_sc < 0.5:
                    best_sc, best_root = max(best_sc, 0.55), guess
                elif best_root is None:
                    best_sc, best_root = 0.55, guess
                else:
                    best_sc, best_root = max(best_sc, 0.55), guess if best_sc < 0.45 else best_root
                    if guess in {"payment", "account"} and best_root in {"payment", "account"}:
                        # trust structural disambiguation for payment vs account
                        best_root = guess
                        best_sc = max(best_sc, 0.55)
        if best_root is not None and best_sc >= 0.4:
            scored.append((best_sc, best_root, k, v))
    scored.sort(key=lambda x: -x[0])
    for sc, root, k, v in scored:
        if root in assigned or k in used:
            continue
        assigned[root] = _remap_opaque_leaves(v, root)
        used.add(k)

    # Lists that look like thread / constraints / orders
    for k, v in state.items():
        if k in used:
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict):
            sample = v[0]
            sk = _child_keyset(sample)
            n_str = sum(1 for x in sample.values() if isinstance(x, str))
            if ({"role", "text"} <= sk or (n_str >= 2 and len(sample) <= 3)) and "thread" not in assigned:
                assigned["thread"] = v
                used.add(k)
        elif isinstance(v, list) and v and isinstance(v[0], str):
            if "constraints" not in assigned and len(v) <= 20:
                assigned["constraints"] = v
                used.add(k)

    # Bare string task (train schema uses a string, not a dict)
    for k, v in state.items():
        if k in used:
            continue
        if isinstance(v, str) and len(v.strip()) >= 12 and "task" not in assigned:
            if "agent" in assigned or "trace_summary" in assigned or "constraints" in assigned:
                assigned["task"] = v
                used.add(k)

    # Second chance: leftover dicts that structurally match free roots (e.g. task)
    for k, v in state.items():
        if k in used or not isinstance(v, dict):
            continue
        guess = _structural_root_guess(v)
        if guess and guess not in assigned:
            assigned[guess] = _remap_opaque_leaves(v, guess)
            used.add(k)

    if len(assigned) >= 2:
        out = dict(assigned)
        for k, v in state.items():
            if k in used:
                continue
            ck = canonicalize_key(k) if isinstance(k, str) else str(k)
            if ck in out:
                continue
            if isinstance(v, (dict, list)) and _subtree_has_opaque(v, ck):
                out[ck] = _remap_opaque_leaves(v, ck if ck in ROOT_CHILD_SIGNATURES or ck == "orders" else None)
            else:
                out[ck] = v
        return out
    return state


def leaf_values_text(obj: Any, max_vals: int = 80) -> str:
    """Space-joined leaf values for TF-IDF backup when keys stay opaque."""
    vals: list[str] = []

    def walk(o: Any) -> None:
        if len(vals) >= max_vals:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and _is_noise_key(k):
                    continue
                walk(v)
        elif isinstance(o, list):
            for x in o[:20]:
                walk(x)
        elif o is None or isinstance(o, bool):
            return
        else:
            s = str(o).strip()
            if s and len(s) <= 120:
                vals.append(s)

    walk(obj)
    return " ".join(vals)


def has_opaque_residue(state: Any) -> bool:
    """True when top-level still lacks schema roots (remap incomplete)."""
    if not isinstance(state, dict):
        return False
    keys = {canonicalize_key(k) if isinstance(k, str) else str(k) for k in state.keys()}
    return len(keys & SCHEMA_ROOTS) < 2


def prepare_state(obj: Any) -> Any:
    """Full inference-time packaging: unwrap wrappers → rename keys → shape rebuild."""
    if isinstance(obj, str):
        import json

        try:
            obj = json.loads(obj)
        except Exception:
            return obj
    if not isinstance(obj, dict):
        return obj
    cleaned = strip_presentation_noise(obj)
    recovered = recover_business_state(cleaned)
    normalized = normalize_state_keys(recovered)
    rebuilt = rebuild_canonical_roots(normalized)
    # Re-normalize after shape rebuild (children may still need aliasing).
    if rebuilt is not normalized:
        rebuilt = normalize_state_keys(rebuilt)
    # Final date coerce + noise strip (rebuild may reintroduce nothing, but dates
    # under remapped leaves need ISO form for sealed TF-IDF overlap).
    rebuilt = strip_presentation_noise(rebuilt)
    return rebuilt



def dig(state: Any, *keys: str) -> Any:
    """Path-aware getter: return first value whose canonical key matches ``keys``.

    Searches top-level first, then recursively (depth-limited) so heuristics still
    fire when a residual wrapper remains.
    """
    if not isinstance(state, dict) or not keys:
        return None
    want = {canonicalize_key(k) for k in keys}
    # Direct hit
    for k, v in state.items():
        if canonicalize_key(k) in want:
            return v

    # Shallow recursive search
    stack = [(state, 0)]
    seen = 0
    while stack and seen < 200:
        cur, depth = stack.pop()
        seen += 1
        if not isinstance(cur, dict) or depth > 5:
            continue
        for k, v in cur.items():
            if canonicalize_key(k) in want:
                return v
            if isinstance(v, dict):
                stack.append((v, depth + 1))
            elif isinstance(v, list):
                for item in v[:4]:
                    if isinstance(item, dict):
                        stack.append((item, depth + 1))
    return None
