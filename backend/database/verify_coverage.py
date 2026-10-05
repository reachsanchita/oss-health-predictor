from connection import get_db

def main():
    db = get_db()
    snapshots_col = db["activity_snapshots"]

    repo_ids = snapshots_col.distinct("repository")
    print(f"{'Repository':35} {'Months':8} {'Earliest':10} {'Latest':10}")
    print("-" * 65)

    for repo_id in sorted(repo_ids):
        months = sorted(snapshots_col.distinct("year_month", {"repository": repo_id}))
        count = len(months)
        earliest = months[0] if months else "-"
        latest = months[-1] if months else "-"
        print(f"{repo_id:35} {count:<8} {earliest:10} {latest:10}")

if __name__ == "__main__":
    main()