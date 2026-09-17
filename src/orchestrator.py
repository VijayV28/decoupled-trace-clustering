"""
BPI2019 — orchestrator.

One entry point, ``run_pipeline``, that executes the full decoupled method from the **raw** XES and
writes every artifact the insights notebook consumes:

    raw XES  ->  behavioural + context features  ->  cluster (behaviour)
             ->  KPI-rank cohorts  ->  attribute cohorts to static context (XGBoost)
             ->  SHAP context drivers  ->  behavioural path explanations

Run it from a notebook or the command line; heavy XES parsing is cached, so reruns are quick.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

import data_modelling as dm
import data_preparation as dp


def _default_repo_root():
    # .../src/orchestrator.py -> the repository root is one level up from src
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def run_pipeline(
    repo_root=None,
    k=6,
    force_reparse=False,
    weighting="balanced",
    shap_sample=15000,
    verbose=True,
):
    """Execute the end-to-end pipeline and persist artifacts. Returns a results dict."""
    repo_root = repo_root or _default_repo_root()
    out = os.path.join(repo_root, "artifacts")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    # 1. data preparation (from raw XES) ------------------------------------- #
    prep = dp.prepare(repo_root, force_reparse=force_reparse, verbose=verbose)
    events, proc, ctx, bigram_cols = (
        prep["events"],
        prep["proc"],
        prep["ctx"],
        prep["bigram_cols"],
    )

    # 2. cluster on behaviour ------------------------------------------------ #
    labels, cl_metrics = dm.cluster_behaviour(proc, k=k)
    if verbose:
        print(
            f"  [model] clustered k={k} | silhouette {cl_metrics['silhouette']:.3f} | "
            f"CH {cl_metrics['calinski_harabasz']:.0f} | DB {cl_metrics['davies_bouldin']:.3f}"
        )

    # 3. KPI-rank the cohorts ----------------------------------------------- #
    kpi = dm.compute_case_kpis(events, proc, labels)
    cohort_kpis = dm.rank_cohorts(kpi)

    # 4. attribute cohorts to static context -------------------------------- #
    model, feats, cats, splits, metrics = dm.train_classifier(
        ctx, labels, weighting=weighting
    )
    Xtr, Xte, ytr, yte = splits
    if verbose:
        print(
            f"  [model] XGBoost [{metrics['weighting']}] acc {metrics['accuracy']:.3f} | "
            f"macro-F1 {metrics['macro_f1']:.3f} | weighted-F1 {metrics['weighted_f1']:.3f}"
        )

    # 5. SHAP feature-value drivers ------------------------------------------ #
    drivers = dm.cluster_value_drivers(
        model, Xte, yte, feats, top_n=6, min_case_pct=20.0
    )

    # 6. behavioural path explanations -------------------------------------- #
    transitions, variants = dm.cohort_path_explanations(
        events, proc, labels, bigram_cols
    )

    # --- persist artifacts -------------------------------------------------- #
    labels_df = proc[[dp.CASE]].copy()
    labels_df["cohort"] = labels
    labels_df.to_csv(os.path.join(out, "cohort_labels.csv"), index=False)
    cohort_kpis.to_csv(os.path.join(out, "cohort_kpis.csv"))
    kpi[
        [
            dp.CASE,
            "cohort",
            "cycle_time_days",
            "automation_ratio",
            "total_events",
            "is_rework",
            "has_payment_block_removed",
        ]
    ].to_csv(os.path.join(out, "case_kpis.csv"), index=False)
    transitions.to_csv(os.path.join(out, "cohort_path_transitions.csv"), index=False)
    variants.to_csv(os.path.join(out, "cohort_top_variants.csv"), index=False)
    drivers.to_csv(os.path.join(out, "shap_cluster_drivers.csv"), index=False)

    per_class = (
        pd.DataFrame(metrics["report"])
        .T.reset_index()
        .rename(columns={"index": "class"})
    )
    per_class.to_csv(os.path.join(out, "classification_report.csv"), index=False)

    summary = {
        "k": int(k),
        "silhouette": round(cl_metrics["silhouette"], 3),
        "calinski_harabasz": round(cl_metrics["calinski_harabasz"], 1),
        "davies_bouldin": round(cl_metrics["davies_bouldin"], 3),
        "n_cases": int(len(labels_df)),
        "cluster_features": dp.ENG_COLS,
        "context_features": feats,
        "context_categorical": cats,
        "note": "structural context only (no vendor); case_value numeric (decile-binned for driver display only); bigram TF-IDF used for explanation only",
        "xgb_params": dm.XGB_PARAMS,
        "classification": {
            "weighting": metrics["weighting"],
            "accuracy": round(metrics["accuracy"], 3),
            "macro_f1": round(metrics["macro_f1"], 3),
            "weighted_f1": round(metrics["weighted_f1"], 3),
        },
        "cohort_kpi_rank": cohort_kpis.reset_index().to_dict("records"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(
            f"  [done] {summary['runtime_seconds']}s | artifacts -> {out}\n"
            "         cohort_labels.csv, cohort_kpis.csv, cohort_path_transitions.csv,\n"
            "         cohort_top_variants.csv, shap_cluster_drivers.csv, classification_report.csv,\n"
            "         pipeline_summary.json"
        )

    return {
        "summary": summary,
        "labels": labels_df,
        "cohort_kpis": cohort_kpis,
        "case_kpis": kpi,
        "metrics": metrics,
        "shap_drivers": drivers,
        "transitions": transitions,
        "variants": variants,
        "model": model,
        "feats": feats,
        "cats": cats,
        "splits": splits,
        "artifacts_dir": out,
    }


if __name__ == "__main__":
    run_pipeline()
