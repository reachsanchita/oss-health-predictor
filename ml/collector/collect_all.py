"""
Phase 4d - Config-driven, resumable collector.

Run from the project root:
    python ml\\collector\\collect_all.py

Reads  data/raw/selected_repos.csv
Writes data/raw/monthly/<repository>.csv   (one small file per repo)
       data/raw/collection_log.csv

Safe to stop (Ctrl+C) and re-run: repos already collected are skipped.

Notes (for your report):
 - commits + active contributors come from /stats/contributors (full history,
   weekly buckets; GitHub returns at most the top ~100 contributors).
 - PRs / issues / releases come from paginated lists, capped at MAX_PAGES*100
   newest items. Older months therefore show 0 = "not collected".
   The labeling code already handles this (coverage rules).
 - Raw commit lists are NOT stored (keeps files small).
"""
import os
import time
import datetime as dt

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise SystemExit("GITHUB_TOKEN not found. Check your .env file.")

API = "https://api.github.com"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
SELECTED_CSV = os.path.join("data", "raw", "selected_repos.csv")
OUT_DIR = os.path.join("data", "raw", "monthly")
LOG_CSV = os.path.join("data", "raw", "collection_log.csv")
MAX_PAGES = 30


def get_json(url, params=None, retries=6):
    """GET with handling for 202 (stats computing), rate limits, and 404/451."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=40)
        except requests.RequestException as e:
            print(f"    network error ({e}); retrying ...")
            time.sleep(5 * (attempt + 1))
            continue

        remaining = r.headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) < 10:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 0) + 5
            print(f"    rate limit nearly used; sleeping {int(wait)}s ...")
            time.sleep(wait)

        if r.status_code == 200:
            return r.json()
        if r.status_code == 204:
            return []
        if r.status_code == 202:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (403, 429):
            time.sleep(60)
            continue
        if r.status_code in (404, 409, 451):
            return None
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def paginate(url, params, max_pages):
    items, capped = [], False
    for page in range(1, max_pages + 1):
        data = get_json(url, {**params, "per_page": 100, "page": page})
        if not data:
            break
        items.extend(data)
        if len(data) < 100:
            break
    else:
        capped = True
    return items, capped


def monthly_counts(timestamps, index):
    s = pd.to_datetime(pd.Series(timestamps, dtype="object"), utc=True, errors="coerce").dropna()
    if s.empty:
        return pd.Series(0, index=index, dtype=int)
    periods = s.dt.tz_localize(None).dt.to_period("M")
    return periods.value_counts().reindex(index, fill_value=0).astype(int)


def collect_repo(full_name, repo_id):
    contribs = get_json(f"{API}/repos/{full_name}/stats/contributors")
    if not contribs:
        return None, "no_commit_stats"

    rows = []
    for i, c in enumerate(contribs):
        author = (c.get("author") or {}).get("login") or f"anon_{i}"
        for w in c.get("weeks", []):
            if w["c"] > 0:
                rows.append((w["w"], w["c"], author))
    if not rows:
        return None, "no_commits"

    cdf = pd.DataFrame(rows, columns=["w", "c", "author"])
    cdf["period"] = pd.to_datetime(cdf["w"], unit="s").dt.to_period("M")
    commits = cdf.groupby("period")["c"].sum()
    contributors = cdf.groupby("period")["author"].nunique()

    current = pd.Period(dt.date.today(), freq="M")
    index = pd.period_range(start=commits.index.min(), end=current, freq="M")

    pulls, pr_capped = paginate(f"{API}/repos/{full_name}/pulls",
                                {"state": "all", "sort": "created", "direction": "desc"}, MAX_PAGES)
    raw_issues, is_capped = paginate(f"{API}/repos/{full_name}/issues",
                                     {"state": "all", "sort": "created", "direction": "desc"}, MAX_PAGES)
    issues = [i for i in raw_issues if "pull_request" not in i]
    releases, _ = paginate(f"{API}/repos/{full_name}/releases", {}, 5)

    df = pd.DataFrame(index=index)
    df["commits_count"] = commits.reindex(index, fill_value=0).astype(int)
    df["active_contributors_count"] = contributors.reindex(index, fill_value=0).astype(int)
    df["prs_opened"] = monthly_counts([p.get("created_at") for p in pulls], index)
    df["prs_merged"] = monthly_counts([p.get("merged_at") for p in pulls], index)
    df["issues_opened"] = monthly_counts([i.get("created_at") for i in issues], index)
    df["issues_closed"] = monthly_counts([i.get("closed_at") for i in issues], index)
    df["releases_count"] = monthly_counts(
        [r.get("published_at") or r.get("created_at") for r in releases], index)

    df.insert(0, "repository", repo_id)
    df.insert(1, "snapshot_date", [p.start_time.date().isoformat() for p in index])
    df["is_partial_month"] = [p == current for p in index]
    df["prs_list_capped"] = pr_capped
    df["issues_list_capped"] = is_capped
    return df.reset_index(drop=True), "ok"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sel = pd.read_csv(SELECTED_CSV)
    print(f"{len(sel)} repos in selection list.\n")

    log_rows = []
    for n, row in enumerate(sel.itertuples(), start=1):
        out_path = os.path.join(OUT_DIR, f"{row.repository}.csv")
        if os.path.exists(out_path):
            print(f"[{n}/{len(sel)}] {row.full_name}: already collected, skipping")
            continue

        print(f"[{n}/{len(sel)}] {row.full_name} ...")
        start = time.time()
        try:
            df, status = collect_repo(row.full_name, row.repository)
            if df is not None:
                df.to_csv(out_path, index=False)
                print(f"    saved {len(df)} months")
            else:
                print(f"    skipped: {status}")
            log_rows.append({"repo": row.full_name, "status": status,
                             "months": 0 if df is None else len(df),
                             "seconds": round(time.time() - start, 1), "error": ""})
        except KeyboardInterrupt:
            print("\nStopped by user. Re-run to resume.")
            break
        except Exception as e:
            print(f"    ERROR: {e}")
            log_rows.append({"repo": row.full_name, "status": "error", "months": 0,
                             "seconds": round(time.time() - start, 1), "error": str(e)[:200]})

        pd.DataFrame(log_rows).to_csv(LOG_CSV, mode="a", index=False,
                                      header=not os.path.exists(LOG_CSV))
        log_rows = []

    print("\nDone. Check data\\raw\\collection_log.csv for any failures, then re-run to retry them.")


if __name__ == "__main__":
    main()