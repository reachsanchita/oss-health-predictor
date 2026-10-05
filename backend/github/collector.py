import os
import time
import json
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

# ---- Setup ----
load_dotenv()  # reads .env file in the project root

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not found. Check your .env file.")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

BASE_URL = "https://api.github.com"

# Repositories to collect for development/testing (Phase 3)
# Chosen to cover different categories, per Section 6 of our plan.
REPOSITORIES = [
    "psf/requests",              # Python
    "pallets/flask",             # Python web framework
    "expressjs/express",         # JavaScript web framework
    "scikit-learn/scikit-learn", # AI/ML
    "microsoft/vscode",          # Developer tools
]

RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# --- Phase 4b: date-bounded collection window ---
# Commits are pulled using GitHub's "since" filter, so coverage is bounded by
# CALENDAR TIME, not by item count. Combined with the /stats/contributors
# endpoint below, this gives reliable monthly coverage even for very
# high-commit-velocity repos (e.g. vscode).
TARGET_MONTHS_BACK = 24
SINCE_DATE = datetime.now(timezone.utc) - timedelta(days=TARGET_MONTHS_BACK * 30)
SINCE_ISO = SINCE_DATE.strftime("%Y-%m-%dT%H:%M:%SZ")


def github_get(url, params=None, max_retries=3):
    """
    GET request to the GitHub API with rate-limit handling and retries.
    """
    params = params or {}
    response = None

    for attempt in range(1, max_retries + 1):
        response = requests.get(url, headers=HEADERS, params=params)

        # Rate limit hit
        if response.status_code == 403 and "rate limit" in response.text.lower():
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait_seconds = max(reset_time - int(time.time()), 5)
            print(f"[RATE LIMIT] Waiting {wait_seconds}s until reset...")
            time.sleep(wait_seconds)
            continue

        if response.status_code == 200:
            return response

        if response.status_code >= 500:
            wait = 2 ** attempt
            print(f"[SERVER ERROR {response.status_code}] Retrying in {wait}s...")
            time.sleep(wait)
            continue

        # 404, 422, etc — don't retry
        print(f"[ERROR {response.status_code}] {url} -> {response.text[:200]}")
        return response

    return response


def paginated_get(url, params=None, max_pages=10):
    """
    Fetch multiple pages from a GitHub list endpoint.
    max_pages controls how deep we go (100 items per page).
    """
    params = dict(params or {})
    params["per_page"] = 100
    all_items = []
    page = 1

    while page <= max_pages:
        params["page"] = page
        response = github_get(url, params=params)

        if response is None or response.status_code != 200:
            break

        items = response.json()
        if not items:
            break

        all_items.extend(items)

        if len(items) < 100:
            break  # last page reached

        page += 1
        time.sleep(0.3)  # be polite to the API, avoid secondary rate limits

    return all_items


def collect_repository_metadata(owner_repo):
    url = f"{BASE_URL}/repos/{owner_repo}"
    response = github_get(url)
    return response.json() if response and response.status_code == 200 else {}


def collect_contributors(owner_repo):
    url = f"{BASE_URL}/repos/{owner_repo}/contributors"
    return paginated_get(url, params={"anon": "false"}, max_pages=3)


def collect_contributor_stats(owner_repo, max_retries=5, wait_seconds=3):
    """
    GitHub's /stats/contributors endpoint returns weekly commit counts per
    contributor for the repository's ENTIRE history, computed server-side
    by GitHub. Unlike the raw /commits endpoint, this is NOT limited by our
    pagination settings -- it gives reliable monthly coverage even for very
    high-commit-velocity repos (e.g. vscode).

    GitHub computes these stats asynchronously on first request; a 202
    response means "still computing", so we retry after a short wait.
    """
    url = f"{BASE_URL}/repos/{owner_repo}/stats/contributors"
    for attempt in range(max_retries):
        response = github_get(url)
        if response is None:
            return []
        if response.status_code == 200:
            return response.json() or []
        if response.status_code == 202:
            print(f"  (contributor stats still computing, retrying in {wait_seconds}s...)")
            time.sleep(wait_seconds)
            continue
        break
    return []


def collect_commits(owner_repo):
    """
    Date-bounded: pulls commits since SINCE_ISO (not just the most recent N).
    Kept mainly for features like 'last commit date' -- monthly commit
    counts/active contributors now come from the more reliable
    /stats/contributors endpoint instead.
    """
    url = f"{BASE_URL}/repos/{owner_repo}/commits"
    return paginated_get(url, params={"since": SINCE_ISO}, max_pages=15)


def collect_pull_requests(owner_repo):
    """
    PRs don't support clean date-range filtering on this endpoint, so this is
    still count-bounded -- extended as far as practical (up to 2,000) to
    approximate the same ~24-month window as commits.
    """
    url = f"{BASE_URL}/repos/{owner_repo}/pulls"
    return paginated_get(url, params={"state": "all", "sort": "created", "direction": "desc"}, max_pages=20)


def collect_issues(owner_repo):
    """
    Same approach as PRs -- count-bounded, extended to approximate the
    commits' date window. GitHub's /issues endpoint also returns pull
    requests, which we filter out below.
    """
    url = f"{BASE_URL}/repos/{owner_repo}/issues"
    raw_items = paginated_get(
        url, params={"state": "all", "sort": "created", "direction": "desc"}, max_pages=20
    )
    return [item for item in raw_items if "pull_request" not in item]


def collect_releases(owner_repo):
    url = f"{BASE_URL}/repos/{owner_repo}/releases"
    return paginated_get(url, max_pages=5)


def collect_repository(owner_repo):
    print(f"\nCollecting data for: {owner_repo}")

    data = {
        "repository": owner_repo,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "collection_since": SINCE_ISO,  # document the date-bound window used
        "metadata": collect_repository_metadata(owner_repo),
    }
    print("  - metadata done")

    data["contributors"] = collect_contributors(owner_repo)
    print(f"  - contributors: {len(data['contributors'])}")

    data["contributor_stats"] = collect_contributor_stats(owner_repo)
    print(f"  - contributor stats: {len(data['contributor_stats'])} contributors with weekly data")

    data["commits"] = collect_commits(owner_repo)
    print(f"  - commits: {len(data['commits'])}")

    data["pull_requests"] = collect_pull_requests(owner_repo)
    print(f"  - pull requests: {len(data['pull_requests'])}")

    data["issues"] = collect_issues(owner_repo)
    print(f"  - issues: {len(data['issues'])}")

    data["releases"] = collect_releases(owner_repo)
    print(f"  - releases: {len(data['releases'])}")

    return data


def save_raw_data(owner_repo, data):
    safe_name = owner_repo.replace("/", "_")
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    filepath = os.path.join(RAW_DATA_DIR, f"{safe_name}_{date_str}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  -> saved to {filepath}")


def main():
    print(f"Collection window: commits since {SINCE_ISO} (~{TARGET_MONTHS_BACK} months back)\n")
    for owner_repo in REPOSITORIES:
        data = collect_repository(owner_repo)
        save_raw_data(owner_repo, data)

    print("\nCollection complete.")


if __name__ == "__main__":
    main()