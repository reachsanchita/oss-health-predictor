import os
import sys
import pandas as pd
from datetime import datetime, timezone

# Allow importing backend.database.connection from this script's location
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "backend", "database"))
from connection import get_db

PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

MIN_RELIABLE_MONTHS = 6  # repos with fewer months than this get flagged


def load_raw_snapshots():
    db = get_db()
    snapshots = list(db["activity_snapshots"].find({}))
    repos = {r["_id"]: r for r in db["repositories"].find({})}
    return snapshots, repos


def to_dataframe(snapshots):
    df = pd.DataFrame(snapshots)
    df = df.drop(columns=["_id"], errors="ignore")
    return df


def clean_dataframe(df):
    # --- Remove exact duplicate (repository, year_month) rows, keep latest ---
    before = len(df)
    df = df.drop_duplicates(subset=["repository", "year_month"], keep="last")
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate snapshot rows.")

    # --- Ensure numeric columns are numeric and missing counts become 0 ---
    numeric_cols = [
        "commits_count", "active_contributors_count", "prs_opened",
        "prs_merged", "issues_opened", "issues_closed", "releases_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # --- Parse dates properly ---
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["year_month"] = df["snapshot_date"].dt.strftime("%Y-%m")

    # --- Sort chronologically within each repository ---
    df = df.sort_values(["repository", "snapshot_date"]).reset_index(drop=True)

    return df


def flag_data_quality(df):
    """
    Adds per-row and per-repo data quality flags.
    Does NOT delete anything -- just labels it, so downstream phases
    (feature engineering, modeling) can decide how to handle it.
    """
    # Flag the most recent month for each repo as partial
    # (it's still in progress at the time of collection)
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    df["is_partial_month"] = df["year_month"] == current_month

    # Count total reliable months per repo
    months_per_repo = df.groupby("repository")["year_month"].nunique()
    short_history_repos = months_per_repo[months_per_repo < MIN_RELIABLE_MONTHS].index.tolist()

    df["short_history_repo"] = df["repository"].isin(short_history_repos)

    return df, short_history_repos


def flag_metric_coverage(df):
    """
    Commits/PRs/issues are pulled from a limited recent window (most recent N items),
    while releases can reach further back in time. This means early months may show
    releases_count > 0 but commits_count == 0 -- that's a DATA GAP, not real zero activity.

    This flags any row before a repo's earliest real commit activity as having
    incomplete metric coverage, so downstream phases can exclude it rather than
    treat "no data" as "zero activity".
    """
    # Earliest month per repo where we actually observed commit activity
    commit_months = df[df["commits_count"] > 0].groupby("repository")["year_month"].min()

    def is_covered(row):
        start = commit_months.get(row["repository"])
        if start is None:
            return False  # repo has no commit data at all -- fully unreliable
        return row["year_month"] >= start

    df["has_full_metric_coverage"] = df.apply(is_covered, axis=1)
    return df


def save_outputs(df, repos):
    # Attach readable repo metadata for convenience in EDA
    df["full_name"] = df["repository"].map(lambda r: repos.get(r, {}).get("full_name", r))

    csv_path = os.path.join(PROCESSED_DIR, "activity_snapshots_clean.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved cleaned dataset -> {csv_path}  ({len(df)} rows)")

    db = get_db()
    processed_col = db["processed_snapshots"]
    processed_col.delete_many({})  # rebuild fresh each time this script runs
    records = df.to_dict("records")
    if records:
        processed_col.insert_many(records)
    print(f"Saved cleaned dataset -> MongoDB.processed_snapshots  ({len(records)} docs)")


def main():
    snapshots, repos = load_raw_snapshots()
    print(f"Loaded {len(snapshots)} raw snapshot documents.")

    df = to_dataframe(snapshots)
    df = clean_dataframe(df)
    df, short_history_repos = flag_data_quality(df)
    df = flag_metric_coverage(df)

    incomplete_count = (~df["has_full_metric_coverage"]).sum()
    print(f"\nRows with incomplete metric coverage (pre-dates real commit data): {incomplete_count} / {len(df)}")

    print("\n--- Summary ---")
    print(f"Total repositories: {df['repository'].nunique()}")
    print(f"Total snapshot rows: {len(df)}")
    print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
    if short_history_repos:
        print(f"Repos flagged as short history (<{MIN_RELIABLE_MONTHS} months): {short_history_repos}")
    else:
        print("No repos flagged as short history -- all repos meet the minimum threshold.")

    print("\nSample rows:")
    print(df[["repository", "year_month", "commits_count", "active_contributors_count",
               "prs_merged", "issues_closed", "releases_count"]].head(10).to_string(index=False))

    save_outputs(df, repos)


if __name__ == "__main__":
    main()