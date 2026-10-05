import os
import json
import glob
from datetime import datetime, timezone
from collections import defaultdict

from connection import get_db

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def parse_iso_date(date_str):
    """Parses GitHub's ISO date strings safely. Returns None if missing/invalid."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def month_key(dt):
    """Returns a 'YYYY-MM' string for bucketing."""
    return dt.strftime("%Y-%m") if dt else None


def build_repository_doc(raw):
    meta = raw.get("metadata", {})
    full_name = meta.get("full_name", raw["repository"])
    owner, name = full_name.split("/") if "/" in full_name else (full_name, "")

    return {
        "_id": full_name.replace("/", "_"),
        "owner": owner,
        "name": name,
        "full_name": full_name,
        "description": meta.get("description"),
        "language": meta.get("language"),
        "created_at": meta.get("created_at"),
        "first_tracked_at": raw.get("collected_at"),
    }


def bucket_commits_from_stats(contributor_stats):
    """
    Converts GitHub's weekly per-contributor stats into monthly totals.
    This is the reliable source for commits_count and active_contributors_count --
    it covers a repo's full history and isn't limited by our pagination settings.
    Returns: { "2026-01": {"commits_count": X, "active_contributors": {...}}, ... }
    """
    buckets = defaultdict(lambda: {"commits_count": 0, "active_contributors": set()})

    for contributor in contributor_stats or []:
        author_login = (contributor.get("author") or {}).get("login") or "unknown"
        for week in contributor.get("weeks", []):
            commits_in_week = week.get("c", 0)
            if commits_in_week <= 0:
                continue
            week_start = datetime.fromtimestamp(week["w"], tz=timezone.utc)
            mk = month_key(week_start)
            buckets[mk]["commits_count"] += commits_in_week
            buckets[mk]["active_contributors"].add(author_login)

    return buckets


def bucket_monthly_activity(raw):
    """
    Groups activity by month. Commits and active contributors come from the
    reliable /stats/contributors data (full history, not pagination-limited).
    PRs, issues, and releases still come from the paginated lists collected
    earlier.
    """
    # Start with reliable commit/contributor buckets
    stats_buckets = bucket_commits_from_stats(raw.get("contributor_stats", []))

    buckets = defaultdict(lambda: {
        "commits_count": 0,
        "active_contributors": set(),
        "prs_opened": 0,
        "prs_merged": 0,
        "issues_opened": 0,
        "issues_closed": 0,
        "releases_count": 0,
    })

    # Merge in the reliable commit/contributor data
    for mk, data in stats_buckets.items():
        buckets[mk]["commits_count"] = data["commits_count"]
        buckets[mk]["active_contributors"] = set(data["active_contributors"])

    # Pull requests
    for pr in raw.get("pull_requests", []):
        opened = parse_iso_date(pr.get("created_at"))
        merged = parse_iso_date(pr.get("merged_at"))
        if opened:
            buckets[month_key(opened)]["prs_opened"] += 1
        if merged:
            buckets[month_key(merged)]["prs_merged"] += 1

    # Issues
    for issue in raw.get("issues", []):
        opened = parse_iso_date(issue.get("created_at"))
        closed = parse_iso_date(issue.get("closed_at"))
        if opened:
            buckets[month_key(opened)]["issues_opened"] += 1
        if closed:
            buckets[month_key(closed)]["issues_closed"] += 1

    # Releases
    for release in raw.get("releases", []):
        published = parse_iso_date(release.get("published_at"))
        if published:
            buckets[month_key(published)]["releases_count"] += 1

    # Convert sets to counts (Mongo can't store Python sets)
    for mk, data in buckets.items():
        data["active_contributors_count"] = len(data["active_contributors"])
        del data["active_contributors"]

    return buckets


def build_snapshot_docs(repo_id, buckets):
    docs = []
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")

    for mk, data in buckets.items():
        docs.append({
            "repository": repo_id,
            "year_month": mk,
            "snapshot_date": f"{mk}-01",
            "snapshot_type": "backfilled",  # honesty: reconstructed from dated events, not a live snapshot
            "is_partial_month": (mk == current_month),  # current month is still in progress
            "commits_count": data["commits_count"],
            "active_contributors_count": data["active_contributors_count"],
            "prs_opened": data["prs_opened"],
            "prs_merged": data["prs_merged"],
            "issues_opened": data["issues_opened"],
            "issues_closed": data["issues_closed"],
            "releases_count": data["releases_count"],
        })

    return docs


def main():
    db = get_db()
    repositories_col = db["repositories"]
    snapshots_col = db["activity_snapshots"]

    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))
    print(f"Found {len(raw_files)} raw data files.\n")

    for filepath in raw_files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        repo_doc = build_repository_doc(raw)
        repo_id = repo_doc["_id"]

        repositories_col.update_one(
            {"_id": repo_id}, {"$set": repo_doc}, upsert=True
        )

        buckets = bucket_monthly_activity(raw)
        snapshot_docs = build_snapshot_docs(repo_id, buckets)

        for doc in snapshot_docs:
            snapshots_col.update_one(
                {"repository": doc["repository"], "year_month": doc["year_month"]},
                {"$set": doc},
                upsert=True,
            )

        print(f"{repo_doc['full_name']}: {len(snapshot_docs)} monthly snapshots built")

    print("\nHistorical dataset build complete.")


if __name__ == "__main__":
    main()