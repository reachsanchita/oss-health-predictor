"""
Phase 4c - Reproducible, stratified repository selection.

Run from the project root:
    python ml\\selection\\select_repos.py

Output:
    data/raw/selected_repos.csv        (the repo list used by the collector)
    reports/tables/selection_log.md    (queries + counts, methodology evidence)

Method: GitHub Search API, strata = activity status x language x stars band,
random sampling with a fixed seed (SEED), round-robin across strata.
"""
import os
import time
import random
import datetime as dt

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise SystemExit("GITHUB_TOKEN not found. Check your .env file (variable name must be GITHUB_TOKEN).")

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
SEED = 42
OUT_CSV = os.path.join("data", "raw", "selected_repos.csv")
LOG_MD = os.path.join("reports", "tables", "selection_log.md")

# Already collected in Phases 4-6 (skip them)
ALREADY_COLLECTED = {
    "expressjs/express", "microsoft/vscode", "pallets/flask",
    "psf/requests", "scikit-learn/scikit-learn",
}

LANGUAGES = ["Python", "JavaScript", "TypeScript", "Java", "Go", "C++", "Rust"]
STAR_BANDS = [(200, 2000), (2000, 20000), (20000, 500000)]
EXCLUDE_WORDS = ("awesome", "tutorial", "example", "dotfiles", "cheatsheet",
                 "interview", "roadmap", "course", "leetcode", "algorithms", "samples")

TODAY = dt.date.today()


def days_ago(n):
    return (TODAY - dt.timedelta(days=n)).isoformat()


MIN_AGE_CUTOFF = days_ago(3 * 365)   # created at least 3 years ago

# status -> (target number of repos, search qualifiers)
STATUS = {
    "active":   (20, f"pushed:>={days_ago(90)} archived:false"),
    "stale":    (10, f"pushed:{days_ago(365)}..{days_ago(180)} archived:false"),
    "dormant":  (12, f"pushed:<{days_ago(365)} archived:false"),
    "archived": (15, "archived:true"),
}


def search_repos(query, rng):
    url = "https://api.github.com/search/repositories"
    page = rng.randint(1, 3)
    for attempt in range(6):
        r = requests.get(url, headers=HEADERS, timeout=30,
                         params={"q": query, "per_page": 100, "page": page,
                                 "sort": "stars", "order": "desc"})
        if r.status_code == 200:
            items = r.json().get("items", [])
            if not items and page != 1:
                page = 1          # that random page was empty, fall back to page 1
                continue
            time.sleep(2.5)       # search API allows ~30 requests/minute
            return items
        if r.status_code in (403, 429):
            print("  rate limited, waiting 60s ...")
            time.sleep(60)
            continue
        if r.status_code == 422:
            return []
        time.sleep(3 * (attempt + 1))
    return []


def keep(item):
    name = (item["full_name"] + " " + (item.get("description") or "")).lower()
    if item["full_name"] in ALREADY_COLLECTED:
        return False
    if item.get("fork") or item.get("mirror_url") or item.get("is_template"):
        return False
    if any(w in name for w in EXCLUDE_WORDS):
        return False
    return True


def sample_status(pool, target, rng, taken):
    """Round-robin random sampling across (language, band) combos."""
    combos = [c for c in pool if pool[c]]
    for c in combos:
        rng.shuffle(pool[c])
    chosen = []
    while len(chosen) < target and any(pool[c] for c in combos):
        rng.shuffle(combos)
        for c in combos:
            if len(chosen) >= target:
                break
            while pool[c]:
                item = pool[c].pop()
                if item["full_name"] not in taken:
                    taken.add(item["full_name"])
                    chosen.append(item)
                    break
    return chosen


def main():
    rng = random.Random(SEED)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_MD), exist_ok=True)

    taken, rows, log = set(), [], []
    log.append(f"# Repository selection log\n\nRun date: {TODAY}  |  seed: {SEED}\n")

    for status, (target, qual) in STATUS.items():
        print(f"\n=== Status: {status} (target {target}) ===")
        pool = {}
        for lang in LANGUAGES:
            for lo, hi in STAR_BANDS:
                q = f"language:{lang} stars:{lo}..{hi} created:<{MIN_AGE_CUTOFF} fork:false {qual}"
                items = [i for i in search_repos(q, rng) if keep(i)]
                pool[(lang, f"{lo}-{hi}")] = items
                log.append(f"- `{q}` -> {len(items)} candidates")
                print(f"  {lang:<11} {lo}-{hi}: {len(items)} candidates")
        chosen = sample_status(pool, target, rng, taken)
        for it in chosen:
            rows.append({
                "full_name": it["full_name"],
                "repository": it["full_name"].replace("/", "_"),
                "language": it.get("language"),
                "stars": it["stargazers_count"],
                "created_at": it["created_at"],
                "pushed_at": it["pushed_at"],
                "archived": it["archived"],
                "status_stratum": status,
                "description": (it.get("description") or "")[:120],
            })
        print(f"  -> selected {len(chosen)} for '{status}'")
        log.append(f"\n**{status}: selected {len(chosen)} of target {target}**\n")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    with open(LOG_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    print(f"\nSaved {len(df)} repos -> {OUT_CSV}")
    print("\nBy status:\n", df["status_stratum"].value_counts().to_string())
    print("\nBy language:\n", df["language"].value_counts().to_string())
    print(f"\nArchived repos: {int(df['archived'].sum())}")


if __name__ == "__main__":
    main()