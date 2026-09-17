"""
BPI2019 — data preparation.

Turns the **raw** BPI Challenge 2019 XES log into the feature tables the pipeline needs:

- ``parse_xes``                 : stream the 695 MB XES into tidy ``events`` / ``cases`` tables
                                  (cached to parquet so reruns are fast).
- ``build_behavioural_features``: 11 engineered control-flow aggregates + bigram TF-IDF transitions
                                  (the behavioural space used for clustering).
- ``build_context_features``    : native-categorical static context (procurement metadata) +
                                  ``case_value_eur_log`` (the excluded space used for attribution).

Feature definitions reproduce exactly the ones validated in notebooks ``01``/``02`` so the pipeline
regenerates the same cohorts. No dimensionality reduction; everything stays interpretable.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- XES field names -------------------------------------------------------- #
CASE, ACT, TS = "case:concept:name", "concept:name", "time:timestamp"
RESOURCE, USER, NETWORTH = "org:resource", "User", "Cumulative net worth (EUR)"
_ATTR_TAGS = {"string", "date", "int", "float", "boolean", "id"}
BG_SEP = " || "  # transition-token separator (never appears inside an activity name)

# --- feature contracts (order matters for reproducibility) ------------------ #
ENG_COLS = [
    "total_events",
    "unique_activities",
    "n_resources",
    "n_goods_receipts",
    "has_payment_block_removed",
    "has_change_activity",
    "has_cancel_activity",
    "has_delete_po_item",
    "automation_ratio",
    "repetition_ratio",
    "handoff_ratio",
]
# native-categorical static context (structural only — vendor/identifiers excluded)
CONTEXT_CATS = [
    "item_category",
    "item_type",
    "spend_area",
    "sub_spend_area",
    "spend_classification",
    "document_type",
    "company",
    "gr_based_inv_verif",
    "goods_receipt",
]
CONTEXT_NUM = ["case_value_eur_log"]

_CAT_MAP = {
    "Item Category": "item_category",
    "Item Type": "item_type",
    "Spend area text": "spend_area",
    "Sub spend area text": "sub_spend_area",
    "Spend classification text": "spend_classification",
    "Document Type": "document_type",
    "Company": "company",
}


# --------------------------------------------------------------------------- #
# Raw XES parsing
# --------------------------------------------------------------------------- #
def _local(tag):
    """Namespace-stripped XML tag name."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) and "}" in tag else tag


def parse_xes(xes_path, events_parquet, cases_parquet, force=False, verbose=True):
    """Stream a large XES file into ``(events, cases)`` DataFrames, cached to parquet.

    A single-pass ``lxml.iterparse`` clears processed traces as it goes so peak memory stays low.
    If both parquet caches exist and ``force`` is False, they are loaded instead of re-parsing.
    """
    if not force and os.path.exists(events_parquet) and os.path.exists(cases_parquet):
        if verbose:
            print("  [prep] loaded cached parquet (set force=True to re-parse the XES)")
        return pd.read_parquet(events_parquet), pd.read_parquet(cases_parquet)

    from lxml import etree

    if verbose:
        print(f"  [prep] parsing raw XES: {xes_path}")
    t0 = time.time()
    ev_case, ev_act, ev_ts, ev_res, ev_user, ev_net = [], [], [], [], [], []
    case_records = []
    context = etree.iterparse(xes_path, events=("end",))
    for _, elem in context:
        if _local(elem.tag) != "trace":
            continue
        case_attrs, trace_events = {}, []
        for child in elem:
            lt = _local(child.tag)
            if lt == "event":
                trace_events.append({a.get("key"): a.get("value") for a in child})
            elif lt in _ATTR_TAGS:
                case_attrs[child.get("key")] = child.get("value")
        case_id = case_attrs.get("concept:name")
        case_records.append(case_attrs)
        for ev in trace_events:
            ev_case.append(case_id)
            ev_act.append(ev.get("concept:name"))
            ev_ts.append(ev.get("time:timestamp"))
            ev_res.append(ev.get("org:resource"))
            ev_user.append(ev.get("User"))
            ev_net.append(ev.get("Cumulative net worth (EUR)"))
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    events = pd.DataFrame(
        {
            CASE: ev_case,
            ACT: ev_act,
            TS: pd.to_datetime(ev_ts, utc=True, errors="coerce"),
            RESOURCE: ev_res,
            USER: ev_user,
            NETWORTH: pd.to_numeric(ev_net, errors="coerce"),
        }
    )
    cases = pd.DataFrame(case_records).rename(columns={"concept:name": CASE})
    os.makedirs(os.path.dirname(events_parquet), exist_ok=True)
    events.to_parquet(events_parquet, index=False)
    cases.to_parquet(cases_parquet, index=False)
    if verbose:
        print(
            f"  [prep] parsed {events[CASE].nunique():,} cases / {len(events):,} events "
            f"in {time.time() - t0:.0f}s (cached)"
        )
    return events, cases


# --------------------------------------------------------------------------- #
# Behavioural (process) features — used for clustering
# --------------------------------------------------------------------------- #
def build_behavioural_features(events, max_tfidf=30):
    """Per-case behaviour: 11 engineered aggregates + <=``max_tfidf`` bigram TF-IDF transitions.

    Returns ``(proc_df, bigram_cols)`` where ``proc_df`` has the case key, the ``ENG_COLS`` and the
    ``bg_*`` TF-IDF columns. Vectorised (no per-group ``apply``) for speed on ~1.6M events.
    """
    ev = events[[CASE, ACT, RESOURCE]].copy()
    ev["is_human"] = ev[RESOURCE].astype(str).str.startswith("user")
    ev["is_gr"] = ev[ACT] == "Record Goods Receipt"
    ev["is_block_removed"] = ev[ACT] == "Remove Payment Block"
    ev["is_change"] = ev[ACT].astype(str).str.startswith("Change")
    ev["is_cancel"] = ev[ACT].astype(str).str.startswith("Cancel")
    ev["is_delete_po"] = ev[ACT] == "Delete Purchase Order Item"

    g = ev.groupby(CASE, sort=False)
    eng = pd.DataFrame(
        {
            "total_events": g.size(),
            "unique_activities": g[ACT].nunique(),
            "n_resources": g[RESOURCE].nunique(),
            "n_goods_receipts": g["is_gr"].sum().astype(int),
            "has_payment_block_removed": g["is_block_removed"].max().astype(int),
            "has_change_activity": g["is_change"].max().astype(int),
            "has_cancel_activity": g["is_cancel"].max().astype(int),
            "has_delete_po_item": g["is_delete_po"].max().astype(int),
            "automation_ratio": (1 - g["is_human"].mean()).round(4),
        }
    )
    eng["repetition_ratio"] = (
        1 - eng["unique_activities"] / eng["total_events"]
    ).round(4)
    eng["handoff_ratio"] = (eng["n_resources"] / eng["total_events"]).round(4)
    eng = eng.reset_index()

    # bigram TF-IDF over "A -> B" activity transitions (events are already in log order)
    tr = events[[CASE, ACT]].copy()
    tr["next"] = tr.groupby(CASE, sort=False)[ACT].shift(-1)
    tr = tr.dropna(subset=["next"])
    tr["trans"] = tr[ACT] + " -> " + tr["next"]
    docs = tr.groupby(CASE, sort=False)["trans"].agg(BG_SEP.join).reset_index()
    docs.columns = [CASE, "transition_doc"]
    docs = (
        pd.DataFrame({CASE: eng[CASE]})
        .merge(docs, on=CASE, how="left")
        .fillna({"transition_doc": ""})
    )
    vec = TfidfVectorizer(
        lowercase=False,
        max_features=max_tfidf,
        tokenizer=lambda t: t.split(BG_SEP) if t else [],
        token_pattern=None,
    )
    tf = vec.fit_transform(docs["transition_doc"])
    bg = pd.DataFrame(
        tf.toarray(), columns=[f"bg_{n}" for n in vec.get_feature_names_out()]
    )
    bg.insert(0, CASE, docs[CASE].values)
    proc = eng.merge(bg, on=CASE, how="left").fillna(0)
    bigram_cols = [c for c in bg.columns if c.startswith("bg_")]
    return proc, bigram_cols


# --------------------------------------------------------------------------- #
# Static context features — the excluded space used for attribution
# --------------------------------------------------------------------------- #
def _truthy(s):
    """Robust 0/1 from true/false/NaN stored as strings or bools."""
    return s.astype(str).str.lower().isin(["true", "1"]).astype(int)


def build_context_features(events, cases):
    """Per-case native-categorical static context + ``case_value_eur_log``.

    Categoricals are returned as raw (stripped) strings for native handling by XGBoost; booleans are
    mapped to ``Yes``/``No``. No one-hot, no truncation here (folding happens at model time).
    """
    cf = cases.set_index(CASE)
    case_value = (
        events.groupby(CASE, sort=False)[NETWORTH].max().rename("case_value_eur")
    )
    ctx = pd.DataFrame(index=cf.index)
    ctx["case_value_eur_log"] = np.log1p(
        case_value.reindex(cf.index).clip(lower=0).fillna(0)
    ).round(4)
    ctx["gr_based_inv_verif"] = np.where(
        _truthy(cf["GR-Based Inv. Verif."]) == 1, "Yes", "No"
    )
    ctx["goods_receipt"] = np.where(_truthy(cf["Goods Receipt"]) == 1, "Yes", "No")
    for src, name in _CAT_MAP.items():
        ctx[name] = (
            cf[src].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})
        )
    return ctx.reset_index()


def fold_high_cardinality(s, cap=12, cover=0.85):
    """Collapse a categorical to its top values covering ``cover`` of the mass (max ``cap``) + Others."""
    vc = s.value_counts()
    if len(vc) <= cap:
        return s
    frac = vc.cumsum() / vc.sum()
    keep = set(vc.index[: min(cap, int((frac.values < cover).sum()) + 1)])
    return s.where(s.isin(keep), "Others")


def prepare(repo_root, force_reparse=False, max_tfidf=30, verbose=True):
    """End-to-end data preparation from the raw XES.

    Returns a dict with ``events``, ``cases``, ``proc`` (behavioural), ``ctx`` (context) and
    ``bigram_cols``.
    """
    interim = os.path.join(repo_root, "data", "interim")
    xes = os.path.join(repo_root, "data", "BPI_Challenge_2019.xes")
    events, cases = parse_xes(
        xes,
        os.path.join(interim, "bpi2019_events.parquet"),
        os.path.join(interim, "bpi2019_cases.parquet"),
        force=force_reparse,
        verbose=verbose,
    )
    if verbose:
        print("  [prep] engineering behavioural + context features ...")
    proc, bigram_cols = build_behavioural_features(events, max_tfidf=max_tfidf)
    ctx = build_context_features(events, cases)
    return {
        "events": events,
        "cases": cases,
        "proc": proc,
        "ctx": ctx,
        "bigram_cols": bigram_cols,
    }
