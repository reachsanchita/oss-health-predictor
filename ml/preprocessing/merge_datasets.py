"""
Merge the original 5-repo dataset with all newly collected repos.

Run from the project root:
    python ml\\preprocessing\\merge_datasets.py

Idempotent: always rebuilds from the backed-up original, so re-running never
creates duplicates.
"""
import os
import glob
import shutil

import pandas as pd

CLEAN_CSV = os.path.join("data", "processed", "activity_snapshots_clean.csv")
BACKUP_CSV = os.path.join("data", "processed", "activity_snapshots_clean_5repos.csv")
MONTHLY_GLOB = os.path.join("data", "raw", "monthly", "*.csv")
MIN_COMPLETE_MONTHS = 30

NUMERIC = ["commits_count", "active_contributors_count", "prs_opened",
           "prs_merged", "issues_opened", "issues_closed", "releases_count"]


def main():
    if not os.path.exists(BACKUP_CSV):
        shutil.copy(CLEAN_CSV, BACKUP_CSV)
        print(f"Backed up original dataset -> {BACKUP_CSV}")

    base = pd.read_csv(BACKUP_CSV)
    print(f"Original: {len(base)} rows, {base['repository'].nunique()} repos")

    frames = []
    skipped = []
    for path in sorted(glob.glob(MONTHLY_GLOB)):
        d = pd.read_csv(path)
        complete = int((~d["is_partial_month"].astype(bool)).sum())
        if complete < MIN_COMPLETE_MONTHS:
            skipped.append((os.path.basename(path), complete))
            continue
        frames.append(d)

    print(f"New repos kept: {len(frames)} | skipped (<{MIN_COMPLETE_MONTHS} months): {len(skipped)}")
    for name, m in skipped:
        print(f"   skipped {name}: {m} months")

    merged = pd.concat([base] + frames, ignore_index=True, sort=False)
    merged["snapshot_date"] = pd.to_datetime(merged["snapshot_date"])
    merged["is_partial_month"] = merged["is_partial_month"].fillna(False).astype(bool)
    for col in NUMERIC:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    merged = (merged.sort_values(["repository", "snapshot_date"])
                    .drop_duplicates(subset=["repository", "snapshot_date"], keep="last"))
    merged.to_csv(CLEAN_CSV, index=False)

    print(f"\nMerged: {len(merged)} rows, {merged['repository'].nunique()} repos -> {CLEAN_CSV}")
    print("\nMonths per repo (min / median / max):",
          merged.groupby("repository").size().agg(["min", "median", "max"]).tolist())


if __name__ == "__main__":
    main()