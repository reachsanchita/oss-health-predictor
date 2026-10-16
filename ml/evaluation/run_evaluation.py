"""
Phase 18 - Research evaluation experiments.

Run:  python -m ml.evaluation.run_evaluation     (after train_models)

1. Ablation by feature group      -> reports/tables/eval_ablation.csv
2. Seed stability (5 seeds)       -> reports/tables/eval_seeds.csv
3. Per language / status / stars  -> reports/tables/eval_by_stratum.csv
4. Label-threshold sensitivity    -> reports/tables/eval_threshold_sensitivity.csv
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from ml.models.common import (load_features, temporal_split, evaluate, fit_model,
                              ARTIFACTS, TABLES, CLASSES)

SELECTED = os.path.join("data", "raw", "selected_repos.csv")
LABELS = os.path.join("data", "labels", "labels.csv")
GROUPS = {
    "trend": ("_slope6", "_ratio3v3"),
    "volatility": ("_std6", "commits_cv6", "commit_zero_months6"),
    "pr_issue": ("prs_", "issues_", "pr_merge_rate", "issue_close_rate", "pr_data", "issue_data"),
    "contributors": ("contributors_", "commits_per_contributor"),
    "releases": ("releases_", "months_since_last_release"),
    "age": ("age_months_log",),
}


def score(model, train, test, cols):
    m = fit_model(clone(model), train[cols], train["y"].to_numpy())
    return evaluate(test["y"].to_numpy(), m.predict(test[cols]), m.predict_proba(test[cols]))


def main():
    bundle = joblib.load(os.path.join(ARTIFACTS, "model.joblib"))
    model, feats = bundle["model"], bundle["features"]
    df, _ = load_features()
    train, test = temporal_split(df)

    # 1. ablation
    rows = [{"removed_group": "none (all features)", "n_features": len(feats), **score(model, train, test, feats)}]
    for g, pats in GROUPS.items():
        cols = [f for f in feats if not any(p in f for p in pats)]
        rows.append({"removed_group": g, "n_features": len(cols), **score(model, train, test, cols)})
    abl = pd.DataFrame(rows)[["removed_group", "n_features", "macro_f1", "balanced_acc", "f1_abandoned"]]
    abl.to_csv(os.path.join(TABLES, "eval_ablation.csv"), index=False)
    print("\nAblation:\n", abl.to_string(index=False))

    # 2. seeds
    srows = []
    for s in range(5):
        m = clone(model).set_params(random_state=s)
        srows.append(score(m, train, test, feats))
    seeds = pd.DataFrame(srows)
    summary = seeds[["macro_f1", "balanced_acc", "f1_abandoned"]].agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(TABLES, "eval_seeds.csv"))
    print("\nSeed stability (5 seeds):\n", summary.to_string())

    # 3. strata (uses the saved temporal-test predictions)
    preds = pd.read_csv(os.path.join(TABLES, "test_predictions.csv"))
    if os.path.exists(SELECTED):
        sel = pd.read_csv(SELECTED)
        sel["stars_band"] = pd.cut(sel["stars"], [0, 2000, 20000, 1e9], labels=["200-2k", "2k-20k", "20k+"])
        preds = preds.merge(sel[["repository", "language", "status_stratum", "stars_band"]],
                            on="repository", how="left").fillna({"language": "original5",
                                                                 "status_stratum": "original5"})
        out = []
        for col in ["language", "status_stratum", "stars_band"]:
            for val, g in preds.groupby(col, observed=True):
                if len(g) >= 30:
                    present = np.unique(g["y"])
                    out.append({"by": col, "group": val, "windows": len(g),
                                "macro_f1": round(f1_score(g["y"], g["pred"], average="macro",
                                                           labels=present, zero_division=0), 4)})
        strata = pd.DataFrame(out)
        strata.to_csv(os.path.join(TABLES, "eval_by_stratum.csv"), index=False)
        print("\nPer-stratum macro-F1 (groups with 30+ windows):\n", strata.to_string(index=False))

    # 4. threshold sensitivity (relabel from saved ratios, same features)
    from ml.labeling import diagnose_thresholds as dt
    lab = pd.read_csv(LABELS)
    trows = []
    for healthy, declining in [(0.8, 0.4), (0.7, 0.3), (0.6, 0.25)]:
        lab["new_label"] = lab.apply(lambda r: dt.classify(r["overall_activity_ratio"],
                                                           r["future_zero_commit_months"],
                                                           healthy, declining), axis=1)
        d2 = df.drop(columns=["y"]).merge(lab[["repository", "prediction_month", "new_label"]],
                                          on=["repository", "prediction_month"])
        d2["y"] = d2["new_label"].map({c: i for i, c in enumerate(CLASSES)})
        tr2, te2 = temporal_split(d2)
        res = score(model, tr2, te2, feats)
        shares = d2["new_label"].value_counts(normalize=True).round(3).to_dict()
        trows.append({"healthy>=": healthy, "declining>=": declining, "class_share": shares,
                      "macro_f1": res["macro_f1"], "balanced_acc": res["balanced_acc"]})
    thr = pd.DataFrame(trows)
    thr.to_csv(os.path.join(TABLES, "eval_threshold_sensitivity.csv"), index=False)
    print("\nThreshold sensitivity:\n", thr.to_string(index=False))


if __name__ == "__main__":
    main()
