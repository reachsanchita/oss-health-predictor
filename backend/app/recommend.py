"""Phase 13 - rule-based recommendations driven by SHAP (features pushing AWAY from Healthy)."""

# (substring of feature name, title, advice)  - first matching rule per feature wins
RULES = [
    ("months_since_last_commit", "Restart regular commits",
     "No commits recently. Schedule a small weekly maintenance task so the project shows steady activity."),
    ("commit_zero_months", "Fill the silent months",
     "Several months had zero commits. Plan a lightweight monthly release or dependency-update routine."),
    ("commits_ratio3v3", "Reverse the commit slowdown",
     "Commit activity is dropping versus the previous quarter. Triage the backlog and pick 3-5 achievable issues."),
    ("commits_slope6", "Reverse the commit slowdown",
     "Commit trend is negative. Break work into small tasks and publish a short roadmap."),
    ("contributors_", "Grow the contributor base",
     "Few or falling active contributors. Label good-first-issues, improve CONTRIBUTING.md and thank new contributors."),
    ("commits_per_contributor", "Spread the workload",
     "A few people do most of the work. Share review duties and onboard a second maintainer."),
    ("pr_merge_rate", "Clear the pull-request backlog",
     "Few PRs are being merged. Run a review sprint and set a target response time."),
    ("prs_", "Respond faster to pull requests",
     "Pull-request activity is weak. Acknowledge PRs within a few days and close stale ones."),
    ("issue_close_rate", "Triage open issues",
     "Issues are opened faster than they are closed. Close duplicates and label by priority."),
    ("issues_", "Re-engage with the issue tracker",
     "Issue activity is low. Pin a roadmap issue and ask the community which problems matter most."),
    ("months_since_last_release", "Publish a maintenance release",
     "No release for a long time. Ship a small patch release to signal the project is alive."),
    ("releases_", "Publish a maintenance release",
     "Release cadence has slowed. Adopt a predictable release schedule."),
    ("commits_cv6", "Stabilise activity",
     "Activity is bursty. Aim for steadier contribution through smaller, regular changes."),
]


def recommend(contribs, max_items=3):
    """contribs: list of {"feature","value","shap"} for the Healthy class. Negative shap = hurting health."""
    out, seen = [], set()
    for c in sorted((c for c in contribs if c["shap"] < 0), key=lambda d: d["shap"]):
        for key, title, advice in RULES:
            if key in c["feature"]:
                if title not in seen:
                    seen.add(title)
                    out.append({"title": title, "advice": advice, "because": c["feature"],
                                "value": round(c["value"], 3), "impact": round(abs(c["shap"]), 4)})
                break
        if len(out) >= max_items:
            break
    return out
