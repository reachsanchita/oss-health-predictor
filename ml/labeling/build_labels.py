
import os
import argparse

import pandas as pd

from ml.labeling import diagnose_thresholds as dt

SELECTED_CSV = os.path.join("data", "raw", "selected_repos.csv")
OUT_CSV = os.path.join("data", "labels", "labels.csv")
MIN_BASELINE_COMMITS = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy", type=float, default=0.70)
    ap.add_argument("--declining", type=float, default=0.30)
    args = ap.parse_args()

    df = dt.load_data()
    windows = []
    for repo_id, repo_df in df.groupby("repository"):
        series = dt.build_contiguous_monthly_series(repo_id, repo_df)
        windows.extend(dt.generate_windows_for_repo(series))
    w = pd.DataFrame(windows)

    total = len(w)
    w = w[w["baseline_commits_avg"] >= MIN_BASELINE_COMMITS].copy()
    print(f"\nWindows generated: {total} | dropped (already inactive baseline): {total - len(w)} | kept: {len(w)}")

    w["label"] = w.apply(
        lambda r: dt.classify(r["overall_activity_ratio"], r["future_zero_commit_months"],
                              args.healthy, args.declining), axis=1)
    w["at_risk"] = (w["label"] != "Healthy").astype(int)
    w["prediction_month"] = w["baseline_end"]

    print(f"\n=== Final labels (healthy>={args.healthy}, declining>={args.declining}) ===")
    counts = w["label"].value_counts()
    for lab in ["Healthy", "Declining", "Abandoned"]:
        n = int(counts.get(lab, 0))
        repos = w.loc[w["label"] == lab, "repository"].nunique()
        print(f"{lab:<10} {n:>6} windows ({n / len(w):.1%}) across {repos} repos")
        if n / len(w) < 0.05:
            print(f"   WARNING: {lab} is under 5% -> consider the binary At-Risk target as headline.")
        if repos < 8:
            print(f"   WARNING: {lab} appears in fewer than 8 repos -> weak generalisation evidence.")

    print("\n=== Label audit: up to 5 random windows per class (inspect by eye) ===")
    cols = ["repository", "prediction_month", "baseline_commits_avg", "future_commits_avg",
            "overall_activity_ratio", "future_zero_commit_months", "label"]
    audit = pd.concat([g.sample(min(5, len(g)), random_state=42) for _, g in w.groupby("label")])
    print(audit[cols].to_string(index=False))

    if os.path.exists(SELECTED_CSV):
        sel = pd.read_csv(SELECTED_CSV)
        archived_ids = set(sel.loc[sel["archived"], "repository"])
        a = w[w["repository"].isin(archived_ids)].sort_values("prediction_month")
        if len(a):
            last = a.groupby("repository").tail(6)
            print("\n=== External validation with GitHub's archived flag ===")
            print(f"Archived repos with windows: {a['repository'].nunique()}")
            print(f"Share of their final 6 windows labelled Declining/Abandoned: {last['at_risk'].mean():.1%}")
            print(f"Share of ALL other windows labelled Declining/Abandoned:    "
                  f"{w[~w['repository'].isin(archived_ids)]['at_risk'].mean():.1%}")
            print("(Report this in your thesis: it validates the activity-based definition.)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    keep = ["repository", "prediction_month", "horizon_start", "horizon_end",
            "overall_activity_ratio", "future_zero_commit_months", "n_metrics_used",
            "baseline_commits_avg", "label", "at_risk"]
    w[keep].to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(w)} labelled windows -> {OUT_CSV}")


if __name__ == "__main__":
    main()