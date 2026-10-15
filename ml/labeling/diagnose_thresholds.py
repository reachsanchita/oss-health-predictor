import numpy as np
import pandas as pd
import os

PROCESSED_CSV = os.path.join("data", "processed", "activity_snapshots_clean.csv")
OUTPUT_DIR = os.path.join("data", "processed")
WINDOWS_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "label_diagnostics_windows.csv")

BASELINE_MONTHS = 6   # months used to compute each repo's own baseline
FEATURE_MONTHS = 3    # last 3 baseline months, reserved as ML feature window later
HORIZON_MONTHS = 3    # future months used ONLY to compute the label

# Starting-point thresholds being TESTED (not final)
HEALTHY_THRESHOLD = 0.70
DECLINING_THRESHOLD = 0.30
MIN_ZERO_COMMIT_MONTHS_FOR_ABANDONED = 2

# Ratio stabilizers
# BASELINE_FLOOR: a baseline average below this is treated as this value when
#   used as a denominator (prevents near-zero baselines exploding the ratio).
# RATIO_CAP: hard ceiling on any single metric ratio.
BASELINE_FLOOR = 1.0
RATIO_CAP = 5.0

NUMERIC_COLS = [
    "commits_count", "active_contributors_count", "prs_opened",
    "prs_merged", "issues_opened", "issues_closed", "releases_count",
]

# Metric handling rules (documented for the report):
#  - COMMITS come from GitHub's /stats/contributors (full history) -> always used.
#    If a repo is dead before AND after, commit ratio = 0 (a dead repo is not healthy).
#  - MERGED PRs and CLOSED ISSUES come from lists capped at the most recent
#    ~2,000 items, so older months are "not collected", not "zero activity".
#    These metrics are EXCLUDED from a window when:
#       (a) the baseline starts before that repo's PR/issue data coverage begins, or
#       (b) the metric is inactive (below floor) in both baseline and future
#           (no signal either way).
#  - The overall ratio is the equal-weight mean of the metric ratios that were used.


def load_data():
    df = pd.read_csv(PROCESSED_CSV)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    # The current (in-progress) month is incomplete and would bias the newest
    # window, so it is excluded from this diagnostic (in memory only; the CSV
    # itself is not modified).
    if "is_partial_month" in df.columns:
        partial_count = int(df["is_partial_month"].sum())
        df = df[~df["is_partial_month"]].copy()
        print(f"Excluded {partial_count} in-progress (partial) month rows from the diagnostic.")
    return df


def build_contiguous_monthly_series(repo_id, repo_df):
    """
    Reindexes one repository onto a contiguous monthly calendar, filling
    months with no recorded activity with zeros, so consecutive rows are
    always consecutive calendar months.
    """
    repo_df = repo_df.sort_values("snapshot_date").copy()
    repo_df["period"] = repo_df["snapshot_date"].dt.to_period("M")
    repo_df = repo_df.drop_duplicates(subset="period", keep="last").set_index("period")

    full_range = pd.period_range(start=repo_df.index.min(), end=repo_df.index.max(), freq="M")
    repo_df = repo_df.reindex(full_range)

    for col in NUMERIC_COLS:
        repo_df[col] = repo_df[col].fillna(0)

    repo_df["repository"] = repo_id
    repo_df["year_month"] = repo_df.index.astype(str)
    return repo_df.reset_index(drop=True)


def safe_ratio(future_avg, baseline_avg, floor=BASELINE_FLOOR, cap=RATIO_CAP):
    """future/baseline with a denominator floor and a ratio cap."""
    was_floored = baseline_avg < floor
    effective_baseline = max(baseline_avg, floor)
    raw_ratio = future_avg / effective_baseline
    was_capped = raw_ratio > cap
    return min(raw_ratio, cap), was_floored, was_capped


def optional_metric_ratio(future_avg, baseline_avg, covered):
    """
    Ratio for PR/issue metrics, which may be excluded.
    Returns (ratio_or_None, status, floored, capped).
    """
    if not covered:
        return None, "excluded_no_coverage", False, False
    if baseline_avg < BASELINE_FLOOR and future_avg < BASELINE_FLOOR:
        return None, "excluded_inactive", False, False
    ratio, floored, capped = safe_ratio(future_avg, baseline_avg)
    return ratio, "used", floored, capped


def first_active_position(series):
    """Position of the first month with any activity (None if never)."""
    positions = np.flatnonzero(series.to_numpy() > 0)
    return int(positions[0]) if len(positions) else None


def classify(ratio, zero_commit_months, healthy=HEALTHY_THRESHOLD, declining=DECLINING_THRESHOLD):
    if ratio < declining and zero_commit_months >= MIN_ZERO_COMMIT_MONTHS_FOR_ABANDONED:
        return "Abandoned"
    if ratio < healthy:
        return "Declining"
    return "Healthy"


def generate_windows_for_repo(repo_df):
    n = len(repo_df)
    pr_start = first_active_position(repo_df["prs_opened"])
    issue_start = first_active_position(repo_df["issues_opened"])
    windows = []

    for i in range(BASELINE_MONTHS, n - HORIZON_MONTHS + 1):
        baseline_slice = repo_df.iloc[i - BASELINE_MONTHS : i]
        feature_slice = repo_df.iloc[i - FEATURE_MONTHS : i]   # reserved for Phase 8
        horizon_slice = repo_df.iloc[i : i + HORIZON_MONTHS]

        baseline_start_pos = i - BASELINE_MONTHS
        pr_covered = pr_start is not None and baseline_start_pos >= pr_start
        issue_covered = issue_start is not None and baseline_start_pos >= issue_start

        b_commits = baseline_slice["commits_count"].mean()
        b_prs = baseline_slice["prs_merged"].mean()
        b_issues = baseline_slice["issues_closed"].mean()
        f_commits = horizon_slice["commits_count"].mean()
        f_prs = horizon_slice["prs_merged"].mean()
        f_issues = horizon_slice["issues_closed"].mean()

        commit_ratio, commit_floored, commit_capped = safe_ratio(f_commits, b_commits)
        pr_ratio, pr_status, pr_floored, pr_capped = optional_metric_ratio(f_prs, b_prs, pr_covered)
        issue_ratio, issue_status, issue_floored, issue_capped = optional_metric_ratio(f_issues, b_issues, issue_covered)

        used_ratios = [commit_ratio] + [r for r in (pr_ratio, issue_ratio) if r is not None]
        overall_ratio = float(np.mean(used_ratios))

        zero_commit_months = int((horizon_slice["commits_count"] == 0).sum())

        windows.append({
            "repository": repo_df["repository"].iloc[0],
            "baseline_start": baseline_slice["year_month"].iloc[0],
            "baseline_end": baseline_slice["year_month"].iloc[-1],
            "feature_window_start": feature_slice["year_month"].iloc[0],
            "feature_window_end": feature_slice["year_month"].iloc[-1],
            "horizon_start": horizon_slice["year_month"].iloc[0],
            "horizon_end": horizon_slice["year_month"].iloc[-1],
            "baseline_commits_avg": round(b_commits, 2),
            "baseline_prs_avg": round(b_prs, 2),
            "baseline_issues_avg": round(b_issues, 2),
            "future_commits_avg": round(f_commits, 2),
            "future_prs_avg": round(f_prs, 2),
            "future_issues_avg": round(f_issues, 2),
            "commit_ratio": round(commit_ratio, 4),
            "pr_ratio": round(pr_ratio, 4) if pr_ratio is not None else np.nan,
            "issue_ratio": round(issue_ratio, 4) if issue_ratio is not None else np.nan,
            "pr_status": pr_status,
            "issue_status": issue_status,
            "n_metrics_used": len(used_ratios),
            "overall_activity_ratio": round(overall_ratio, 4),
            "future_zero_commit_months": zero_commit_months,
            "proposed_label": classify(overall_ratio, zero_commit_months),
            "commit_baseline_floored": commit_floored,
            "commit_ratio_capped": commit_capped,
            "pr_ratio_capped": pr_capped,
            "issue_ratio_capped": issue_capped,
        })

    return windows


def print_diagnostics(windows_df):
    print(f"\nTotal valid windows generated: {len(windows_df)}")

    print("\n--- Stabilizer activity ---")
    print(f"commit ratio: baseline floored in {windows_df['commit_baseline_floored'].mean() * 100:.1f}% "
          f"of windows, ratio capped in {windows_df['commit_ratio_capped'].mean() * 100:.1f}%")
    for metric in ["pr", "issue"]:
        shares = windows_df[f"{metric}_status"].value_counts(normalize=True).mul(100).round(1).to_dict()
        capped = windows_df[f"{metric}_ratio_capped"].mean() * 100
        print(f"{metric} ratio: status share (%) = {shares}; capped in {capped:.1f}% of windows")

    print("\n--- Number of metrics combined per window ---")
    print(windows_df["n_metrics_used"].value_counts().sort_index().to_string())

    print("\n--- Overall activity ratio: distribution (all repos combined) ---")
    ratio = windows_df["overall_activity_ratio"]
    print(f"min:    {ratio.min():.4f}")
    print(f"max:    {ratio.max():.4f}")
    print(f"mean:   {ratio.mean():.4f}")
    print(f"median: {ratio.median():.4f}")
    print("\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p:>3}th percentile: {np.percentile(ratio, p):.4f}")

    print("\n--- Windows per repository ---")
    print(windows_df["repository"].value_counts().to_string())

    print("\n--- Overall activity ratio stats by repository ---")
    repo_stats = windows_df.groupby("repository")["overall_activity_ratio"].agg(
        ["count", "min", "max", "mean", "median"]
    ).round(4)
    print(repo_stats.to_string())

    print(f"\n--- Proposed label distribution using {HEALTHY_THRESHOLD:.0%}/{DECLINING_THRESHOLD:.0%} thresholds ---")
    print(windows_df["proposed_label"].value_counts().to_string())

    print("\n--- Proposed label distribution by repository ---")
    label_by_repo = windows_df.groupby(["repository", "proposed_label"]).size().unstack(fill_value=0)
    print(label_by_repo.to_string())

    print("\n--- Threshold sensitivity (number of windows per label) ---")
    rows = []
    for healthy, declining in [(0.8, 0.4), (0.7, 0.3), (0.6, 0.25), (0.5, 0.2)]:
        labels = windows_df.apply(
            lambda r: classify(r["overall_activity_ratio"], r["future_zero_commit_months"], healthy, declining),
            axis=1,
        )
        counts = labels.value_counts()
        rows.append({
            "healthy_if>=": healthy,
            "declining_if>=": declining,
            "Healthy": int(counts.get("Healthy", 0)),
            "Declining": int(counts.get("Declining", 0)),
            "Abandoned": int(counts.get("Abandoned", 0)),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    df = load_data()
    print(f"Loaded {len(df)} processed snapshot rows across {df['repository'].nunique()} repositories.\n")

    all_windows = []
    for repo_id, repo_df in df.groupby("repository"):
        contiguous_df = build_contiguous_monthly_series(repo_id, repo_df)
        repo_windows = generate_windows_for_repo(contiguous_df)
        all_windows.extend(repo_windows)
        print(f"{repo_id}: {len(contiguous_df)} contiguous months -> {len(repo_windows)} valid windows")

    windows_df = pd.DataFrame(all_windows)
    if windows_df.empty:
        print("\nNo valid windows could be generated.")
        return

    print_diagnostics(windows_df)

    windows_df.to_csv(WINDOWS_OUTPUT_CSV, index=False)
    print(f"\nSaved window-level diagnostic data -> {WINDOWS_OUTPUT_CSV}  ({len(windows_df)} rows)")
    print("\nNOTE: DIAGNOSTIC labels only. No official Healthy/Declining/Abandoned labels exist yet.")


if __name__ == "__main__":
    main()