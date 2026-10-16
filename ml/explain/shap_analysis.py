"""
Phase 12 - Explainability.

Run:  python -m ml.explain.shap_analysis      (after train_models)

Writes to reports/figures: shap_bar_<class>.png, shap_beeswarm_Abandoned.png,
shap_dependence_*.png, case_<name>.png  and  reports/tables/case_studies.md
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from ml.models.common import load_features, temporal_split, ARTIFACTS, TABLES, FIGURES, CLASSES
from ml.explain.shap_utils import class_shap, explainer_for, top_contributions


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, name), dpi=200, bbox_inches="tight")
    plt.close()


def case_figure(df, feats, explainer, row, class_idx, name, title):
    x = df.loc[[row]][feats]
    contrib = top_contributions(explainer, x, class_idx)[:8][::-1]
    plt.figure(figsize=(7, 4))
    colors = ["#d64545" if c["shap"] < 0 else "#2e9e5b" for c in contrib]
    plt.barh([f"{c['feature']} = {c['value']:.2f}" for c in contrib], [c["shap"] for c in contrib], color=colors)
    plt.axvline(0, color="k", lw=0.8)
    plt.title(title, fontsize=10)
    plt.xlabel(f"SHAP value (push toward {CLASSES[class_idx]})")
    save(name)
    return contrib


def main():
    bundle = joblib.load(os.path.join(ARTIFACTS, "model.joblib"))
    model, feats = bundle["model"], bundle["features"]
    df, _ = load_features()
    _, test = temporal_split(df)
    sample = test.sample(min(500, len(test)), random_state=42)[feats]

    explainer = explainer_for(model)
    sv = explainer.shap_values(sample)

    mean_abs = []
    for i, c in enumerate(CLASSES):
        sc = class_shap(sv, i)
        mean_abs.append(np.abs(sc).mean(axis=0))
        shap.summary_plot(sc, sample, plot_type="bar", show=False, max_display=15)
        plt.title(f"Top features for class: {c}")
        save(f"shap_bar_{c}.png")

    ab = class_shap(sv, 2)
    shap.summary_plot(ab, sample, show=False, max_display=15)
    plt.title("SHAP beeswarm: Abandoned class")
    save("shap_beeswarm_Abandoned.png")

    overall = pd.DataFrame({"feature": feats, "mean_abs_shap": np.mean(mean_abs, axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    overall.to_csv(os.path.join(TABLES, "shap_global_importance.csv"), index=False)
    for f in overall["feature"].head(5):
        shap.dependence_plot(f, ab, sample, show=False)
        save(f"shap_dependence_{f}.png")

    # three case studies from the temporal test predictions
    preds = pd.read_csv(os.path.join(TABLES, "test_predictions.csv"))
    key = df.set_index(["repository", "prediction_month"]).index
    cases = {
        "healthy_correct": preds[(preds.y == 0) & (preds.pred == 0)],
        "abandoned_correct": preds[(preds.y == 2) & (preds.pred == 2)],
        "false_alarm": preds[(preds.y == 0) & (preds.pred != 0)],
    }
    lines = ["# Case studies (temporal test set)\n"]
    for name, sub in cases.items():
        if sub.empty:
            lines.append(f"## {name}\nNo example found.\n")
            continue
        r = sub.sample(1, random_state=42).iloc[0]
        idx = df.index[(df.repository == r.repository) & (df.prediction_month == r.prediction_month)][0]
        cls = int(r.pred)
        contrib = case_figure(df, feats, explainer, idx, cls,
                              f"case_{name}.png", f"{name}: {r.repository} @ {r.prediction_month}")
        lines.append(f"## {name}\n- Repo: {r.repository}, prediction month {r.prediction_month}\n"
                     f"- True: {CLASSES[int(r.y)]}, predicted: {CLASSES[cls]}\n- Top drivers:")
        for c in contrib[::-1][:5]:
            lines.append(f"  - {c['feature']} = {c['value']:.2f} (SHAP {c['shap']:+.3f})")
        lines.append("")
    with open(os.path.join(TABLES, "case_studies.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("SHAP figures saved to reports/figures; case studies -> reports/tables/case_studies.md")


if __name__ == "__main__":
    main()
