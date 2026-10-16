"""
Phase 8 - Feature engineering.

Run:  python -m ml.features.build_features

Reads  data/labels/labels.csv  +  data/processed/activity_snapshots_clean.csv
Writes data/features/features.csv, data/features/feature_names.json

RULE: every feature uses ONLY months up to and including prediction_month.
The same function (features_for_position) is used by the live backend, so
training and serving can never drift apart.
"""
import os
import json

import numpy as np
import pandas as pd

LABELS_CSV = os.path.join("data", "labels", "labels.csv")
OUT_CSV = os.path.join("data", "features", "features.csv")
NAMES_JSON = os.path.join("data", "features", "feature_names.json")

WINDOW = 6
METRICS = {
    "commits": "commits_count",
    "contributors": "active_contributors_count",
    "prs_opened": "prs_opened",
    "prs_merged": "prs_merged",
    "issues_opened": "issues_opened",
    "issues_closed": "issues_closed",
    "releases": "releases_count",
}


def _slope(y):
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def _first_active(arr):
    pos = np.flatnonzero(np.asarray(arr) > 0)
    return int(pos[0]) if len(pos) else None


def features_for_position(series, p):
    """
    series: contiguous monthly DataFrame for ONE repo (sorted, one row per month)
    p:      integer position of the last month that may be used (inclusive)
    Returns dict of features, or None if there is not enough history.
    """
    if p < WINDOW - 1:
        return None
    win = series.iloc[p - WINDOW + 1: p + 1]
    f = {}

    for short, col in METRICS.items():
        y = win[col].to_numpy(dtype=float)
        f[f"{short}_last"] = y[-1]
        f[f"{short}_mean3"] = y[-3:].mean()
        f[f"{short}_mean6"] = y.mean()
        f[f"{short}_std6"] = y.std()
        f[f"{short}_slope6"] = _slope(y)
        f[f"{short}_ratio3v3"] = (y[-3:].mean() + 1) / (y[:3].mean() + 1)

    c = win["commits_count"].to_numpy(dtype=float)
    contrib = win["active_contributors_count"].to_numpy(dtype=float)
    trailing = 0
    for v in c[::-1]:
        if v == 0:
            trailing += 1
        else:
            break
    f["commit_zero_months6"] = int((c == 0).sum())
    f["months_since_last_commit"] = trailing
    f["commits_cv6"] = c.std() / c.mean() if c.mean() > 0 else 0.0
    f["pr_merge_rate"] = win["prs_merged"].sum() / (win["prs_opened"].sum() + 1)
    f["issue_close_rate"] = win["issues_closed"].sum() / (win["issues_opened"].sum() + 1)
    f["commits_per_contributor"] = c[-3:].mean() / max(contrib[-3:].mean(), 1.0)

    hist = series.iloc[: p + 1]                      # history only, never the future
    rel = np.flatnonzero(hist["releases_count"].to_numpy() > 0)
    f["months_since_last_release"] = float(min(p - rel[-1], 60)) if len(rel) else 60.0
    f["age_months_log"] = float(np.log1p(p + 1))

    window_start = p - WINDOW + 1
    pr_first = _first_active(hist["prs_opened"])
    is_first = _first_active(hist["issues_opened"])
    f["pr_data_covered"] = int(pr_first is not None and window_start >= pr_first)
    f["issue_data_covered"] = int(is_first is not None and window_start >= is_first)
    return f


def main():
    from ml.labeling import diagnose_thresholds as dt

    labels = pd.read_csv(LABELS_CSV)
    df = dt.load_data()

    series = {}
    for repo_id, repo_df in df.groupby("repository"):
        s = dt.build_contiguous_monthly_series(repo_id, repo_df)
        series[repo_id] = (s, {ym: i for i, ym in enumerate(s["year_month"])})

    rows, skipped = [], 0
    for r in labels.itertuples():
        if r.repository not in series:
            skipped += 1
            continue
        s, pos = series[r.repository]
        p = pos.get(r.prediction_month)
        feats = features_for_position(s, p) if p is not None else None
        if feats is None:
            skipped += 1
            continue
        rows.append({"repository": r.repository, "prediction_month": r.prediction_month,
                     "label": r.label, "at_risk": r.at_risk, **feats})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    names = [c for c in out.columns if c not in ("repository", "prediction_month", "label", "at_risk")]
    with open(NAMES_JSON, "w") as fh:
        json.dump(names, fh, indent=2)

    print(f"Feature rows: {len(out)} (skipped {skipped}) | features: {len(names)} | repos: {out['repository'].nunique()}")
    print(out["label"].value_counts().to_string())
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
