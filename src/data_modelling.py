"""
BPI2019 Main — data modelling.

The decoupled method, as reusable functions:

- ``cluster_behaviour``      : z-score the 11 engineered features and fit KMeans (k=6) on behaviour.
- ``compute_case_kpis``      : per-case KPIs (cycle time, automation, rework, cost).
- ``rank_cohorts``           : cohort-level KPI table + an efficiency ranking (best/worst benchmark).
- ``train_classifier``       : tuned native-categorical XGBoost attributing cohorts to static context.
- ``cluster_value_drivers``  : per-cohort feature-value drivers via XGBoost SHAP contributions.
- ``cohort_path_explanations``: distinctive activity transitions + top trace variants per cohort.

Clustering uses **no** dimensionality reduction and the bigram TF-IDF is kept for *explanation*
only (it degrades the clustering distance). The classifier stays on **static context only** to
preserve the decoupling (behaviour must be attributed to context, not peeked at).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from data_preparation import (
    ACT,
    CASE,
    CONTEXT_CATS,
    CONTEXT_NUM,
    ENG_COLS,
    TS,
    fold_high_cardinality,
)

SEED = 42
SIL_SAMPLE = 20000
# tuned in notebook 02 (RandomizedSearchCV, scored on weighted-F1)
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.85,
    colsample_bytree=0.7,
    min_child_weight=1,
    reg_lambda=3,
    gamma=0.0,
)


# --------------------------------------------------------------------------- #
# 1. Cluster on behaviour
# --------------------------------------------------------------------------- #
def cluster_behaviour(proc, k=6, seed=SEED):
    """Z-score the engineered behavioural features and fit KMeans.

    Returns ``(labels, metrics)`` where ``metrics`` holds the silhouette (subsampled),
    Calinski-Harabasz and Davies-Bouldin scores.
    """
    X = StandardScaler().fit_transform(proc[ENG_COLS].astype(float))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    metrics = {
        "silhouette": float(
            silhouette_score(
                X, km.labels_, sample_size=min(SIL_SAMPLE, len(X)), random_state=seed
            )
        ),
        "calinski_harabasz": float(calinski_harabasz_score(X, km.labels_)),
        "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
    }
    return km.labels_, metrics


# --------------------------------------------------------------------------- #
# 2. KPIs + cohort ranking
# --------------------------------------------------------------------------- #
def compute_case_kpis(events, proc, labels):
    """Per-case KPIs: cycle time (days), automation, rework flag, cost, size."""
    span = events.groupby(CASE, sort=False)[TS].agg(["min", "max"])
    cyc = ((span["max"] - span["min"]).dt.total_seconds() / 86400.0).rename(
        "cycle_time_days"
    )
    k = proc[
        [
            CASE,
            "automation_ratio",
            "total_events",
            "has_cancel_activity",
            "has_change_activity",
            "has_delete_po_item",
            "has_payment_block_removed",
        ]
    ].copy()
    k["cohort"] = labels
    k = k.merge(cyc, left_on=CASE, right_index=True, how="left")
    k["is_rework"] = (
        (k.has_cancel_activity > 0)
        | (k.has_change_activity > 0)
        | (k.has_delete_po_item > 0)
    ).astype(int)
    return k


def rank_cohorts(kpi):
    """Aggregate KPIs to cohort level and add an efficiency rank (1 = reference/benchmark).

    Cycle time uses the **median** because BPI2019 has known timestamp anomalies (extreme maxima).
    Efficiency = fast + automated + low rework.
    """
    agg = kpi.groupby("cohort").agg(
        n_cases=("cohort", "size"),
        cycle_time_days_median=("cycle_time_days", "median"),
        cycle_time_days_mean=("cycle_time_days", "mean"),
        automation_rate=("automation_ratio", "mean"),
        rework_rate=("is_rework", "mean"),
        payment_block_rate=("has_payment_block_removed", "mean"),
        avg_events=("total_events", "mean"),
    )
    agg["pct"] = 100 * agg.n_cases / agg.n_cases.sum()
    agg["eff_score"] = (
        agg.cycle_time_days_median.rank()
        + agg.rework_rate.rank()
        + (-agg.automation_rate).rank()
    )
    agg["eff_rank"] = agg.eff_score.rank(method="min").astype(int)
    return agg.sort_values("eff_rank").round(3)


# --------------------------------------------------------------------------- #
# 3. Attribute cohorts to static context
# --------------------------------------------------------------------------- #
def bin_case_value(log_value, n_bins=10):
    """Decile-bin ``case_value_eur_log`` into human-readable EUR-range bands (categorical)."""
    binned, edges = pd.qcut(log_value, q=n_bins, duplicates="drop", retbins=True)
    eur = np.expm1(edges)
    labels = [
        f"D{i + 1} (€{eur[i]:,.0f}–€{eur[i + 1]:,.0f})" for i in range(len(eur) - 1)
    ]
    return binned.cat.rename_categories(labels).astype(str)


def train_classifier(ctx, labels, params=None, seed=SEED, weighting="balanced"):
    """Train a native-categorical XGBoost predicting cohort from static context.

    ``weighting='balanced'`` uses sqrt-balanced sample weights (covers rare cohorts, best for driver
    analysis); ``weighting=None`` optimises overall accuracy. Returns
    ``(model, feats, cats, splits, metrics)``.
    """
    params = params or XGB_PARAMS
    df = ctx.copy()
    df["cohort"] = labels
    cats = list(CONTEXT_CATS)
    for c in cats:
        df[c] = fold_high_cardinality(df[c].astype(str))
    feats = cats + CONTEXT_NUM  # numeric case value kept as-is for the classifier
    y = df["cohort"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        df[feats], y, test_size=0.3, random_state=seed, stratify=y
    )
    for c in cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        random_state=seed,
        eval_metric="mlogloss",
        **params,
    )
    sw = (
        np.sqrt(compute_sample_weight("balanced", ytr))
        if weighting == "balanced"
        else None
    )
    model.fit(Xtr, ytr, sample_weight=sw)
    p = model.predict(Xte)
    metrics = {
        "weighting": weighting or "unweighted",
        "accuracy": float(accuracy_score(yte, p)),
        "macro_f1": float(f1_score(yte, p, average="macro")),
        "weighted_f1": float(f1_score(yte, p, average="weighted")),
        "report": classification_report(yte, p, output_dict=True, zero_division=0),
    }
    return model, feats, cats, (Xtr, Xte, ytr, yte), metrics


# --------------------------------------------------------------------------- #
# 4. SHAP feature-value drivers per cohort
# --------------------------------------------------------------------------- #
def cluster_value_drivers(model, Xte, yte, feats, top_n=6, min_case_pct=20.0):
    """Per-cohort **feature-value** drivers from XGBoost SHAP contributions.

    Mirrors the production driver logic: keep only correctly-classified cases, take each case's SHAP
    toward its (correct) cohort, group by ``(feature, value)``, and rank the value combinations with
    the highest **positive mean SHAP** that occur in at least ``min_case_pct``% of the cohort.
    Numeric context features (e.g. the monetary amount) are decile-binned into readable value bands
    for grouping/display only — the classifier itself is trained on the raw numeric value.
    """
    yhat = model.predict(Xte)
    correct = yhat == yte.to_numpy()
    Xc = Xte[correct].reset_index(drop=True)
    yc = yhat[correct]
    dmat = xgb.DMatrix(Xc, enable_categorical=True)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    if contribs.ndim == 2:  # safety: binary/edge case
        contribs = contribs[:, None, :]
    # display values: numeric features are decile-binned into readable bands; categoricals as-is
    disp_vals, disp_name = {}, {}
    for feat in feats:
        if feat in CONTEXT_NUM:
            disp_vals[feat] = bin_case_value(Xc[feat]).to_numpy()
            disp_name[feat] = feat.replace("_eur_log", "").replace("_log", "") + "_band"
        else:
            disp_vals[feat] = Xc[feat].astype(str).to_numpy()
            disp_name[feat] = feat
    out = []
    for cls in sorted(np.unique(yc)):
        mask = yc == cls
        m = int(mask.sum())
        if m == 0:
            continue
        shp = contribs[mask][:, int(cls), :-1]  # SHAP toward the (correct) cohort
        recs = []
        for j, feat in enumerate(feats):
            g = pd.DataFrame({"value": disp_vals[feat][mask], "mean_shap": shp[:, j]})
            agg = (
                g.groupby("value")
                .agg(mean_shap=("mean_shap", "mean"), count=("mean_shap", "size"))
                .reset_index()
            )
            agg.insert(0, "feature", disp_name[feat])
            recs.append(agg)
        prof = pd.concat(recs, ignore_index=True)
        prof["cohort_pct"] = (100 * prof["count"] / m).round(1)
        prof = prof[(prof.mean_shap > 0) & (prof.cohort_pct >= min_case_pct)]
        prof = prof.nlargest(top_n, "mean_shap").reset_index(drop=True)
        prof.insert(0, "rank", range(1, len(prof) + 1))
        prof.insert(0, "cohort", int(cls))
        out.append(prof)
    drivers = pd.concat(out, ignore_index=True)
    drivers["mean_shap"] = drivers["mean_shap"].round(4)
    return drivers[
        ["cohort", "rank", "feature", "value", "mean_shap", "count", "cohort_pct"]
    ]


# --------------------------------------------------------------------------- #
# 5. Behavioural explanation (bigram lift + trace variants)
# --------------------------------------------------------------------------- #
def cohort_path_explanations(events, proc, labels, bigram_cols, top_n=5):
    """Distinctive activity transitions (bigram lift) + top trace variants per cohort."""
    B = proc.copy()
    B["cohort"] = labels
    gmean = B[bigram_cols].mean()
    tr_rows = []
    for c in sorted(B["cohort"].unique()):
        m = B.loc[B.cohort == c, bigram_cols].mean()
        d = pd.DataFrame({"cohort_mean": m, "global_mean": gmean})
        d["lift"] = (d.cohort_mean + 1e-9) / (d.global_mean + 1e-9)
        d = d[d.cohort_mean > 0.02].sort_values("lift", ascending=False).head(top_n)
        for name, r in d.iterrows():
            tr_rows.append(
                {
                    "cohort": int(c),
                    "transition": name[3:],
                    "cohort_mean": round(float(r.cohort_mean), 4),
                    "global_mean": round(float(r.global_mean), 4),
                    "lift": round(float(r.lift), 2),
                }
            )
    transitions = pd.DataFrame(tr_rows)

    seq = events.groupby(CASE, sort=False)[ACT].agg(" -> ".join).rename("variant")
    vdf = (
        pd.DataFrame(seq).reset_index().merge(B[[CASE, "cohort"]], on=CASE, how="inner")
    )
    var_rows = []
    for c in sorted(vdf.cohort.unique()):
        sub = vdf[vdf.cohort == c]
        for v, cnt in sub.variant.value_counts().head(top_n).items():
            var_rows.append(
                {
                    "cohort": int(c),
                    "pct": round(100 * cnt / len(sub), 1),
                    "n_events": v.count("->") + 1,
                    "variant": v,
                }
            )
    variants = pd.DataFrame(var_rows)
    return transitions, variants
