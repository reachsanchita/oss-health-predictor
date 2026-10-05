import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROCESSED_CSV = os.path.join("data", "processed", "activity_snapshots_clean.csv")
OUTPUT_DIR = os.path.join("notebooks", "data_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(PROCESSED_CSV)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    total_rows = len(df)
    df = df[df["has_full_metric_coverage"] == True].copy()
    print(f"Loaded {total_rows} total rows; using {len(df)} rows with full metric coverage for EDA.")

    return df


def plot_metric_over_time(df, metric, title, filename):
    fig = px.line(
        df, x="snapshot_date", y=metric, color="full_name",
        title=title, labels={"snapshot_date": "Month", metric: metric.replace("_", " ").title()},
        markers=True,
    )
    fig.update_layout(template="plotly_white", legend_title_text="Repository")
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(out_path)
    print(f"Saved: {out_path}")


def plot_correlation_heatmap(df):
    numeric_cols = [
        "commits_count", "active_contributors_count", "prs_opened",
        "prs_merged", "issues_opened", "issues_closed", "releases_count",
    ]
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Between Activity Metrics", template="plotly_white")
    out_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.html")
    fig.write_html(out_path)
    print(f"Saved: {out_path}")


def print_summary_stats(df):
    print("\n--- Summary statistics (full-coverage rows only) ---")
    numeric_cols = [
        "commits_count", "active_contributors_count", "prs_opened",
        "prs_merged", "issues_opened", "issues_closed", "releases_count",
    ]
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n--- Rows per repository (full-coverage only) ---")
    print(df["full_name"].value_counts().to_string())


def main():
    df = load_data()

    print_summary_stats(df)

    plot_metric_over_time(df, "commits_count", "Monthly Commit Activity by Repository", "commits_trend.html")
    plot_metric_over_time(df, "active_contributors_count", "Monthly Active Contributors by Repository", "contributors_trend.html")
    plot_metric_over_time(df, "prs_merged", "Monthly Merged Pull Requests by Repository", "prs_trend.html")
    plot_metric_over_time(df, "issues_closed", "Monthly Closed Issues by Repository", "issues_trend.html")
    plot_metric_over_time(df, "releases_count", "Monthly Releases by Repository", "releases_trend.html")

    plot_correlation_heatmap(df)

    print(f"\nAll charts saved to: {OUTPUT_DIR}")
    print("Open the .html files in your browser to view interactive charts.")


if __name__ == "__main__":
    main()