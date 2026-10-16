"""
Phases 9-10 - Baselines, Random Forest, XGBoost.

Run:  python -m ml.models.train_models

Protocol A (headline): temporal split with a 3-month gap + forward-chaining CV tuning.
Protocol B (generalisation): 5-fold GroupKFold by repository (unseen repos).

Writes:
  reports/tables/model_comparison.csv
  reports/tables/test_predictions.csv
  reports/tables/feature_importance.csv
  reports/figures/confusion_matrix_best.png
  ml/models/artifacts/model.joblib   (model + feature list + classes + params)
  ml/models/artifacts/metrics.json
"""
import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from ml.models.common import (load_features, temporal_split, temporal_cv_splits, evaluate,
                              make_lr, make_rf, make_xgb, fit_model,
                              CLASSES, ARTIFACTS, TABLES, FIGURES, SEED)

RF_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 8, 12, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", 0.3, 0.5],
}
XGB_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 5, 10],
}


class RuleBaseline:
    """Simple trend rule a maintainer might use: compare last 3 months with the 3 before."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        r = X["commits_ratio3v3"].to_numpy()
        out = np.zeros(len(X), dtype=int)
        out[r < 0.7] = 1
        out[r < 0.3] = 2
        return out

    def predict_proba(self, X):
        return np.eye(3)[self.predict(X)]


def main():
    df, feats = load_features()
    train, test = temporal_split(df)
    Xtr, ytr = train[feats], train["y"].to_numpy()
    Xte, yte = test[feats], test["y"].to_numpy()

    print(f"Total windows: {len(df)} | train: {len(train)} | test: {len(test)} (3-month gap removed)")
    print("Train classes:", train["label"].value_counts().to_dict())
    print("Test classes: ", test["label"].value_counts().to_dict())
    if len(set(yte)) < 3:
        print("WARNING: test set is missing a class; metrics will be less reliable.")

    rows = []

    def add(name, protocol, metrics):
        rows.append({"model": name, "protocol": protocol, **metrics})
        print(f"  {name:<22} macro-F1 {metrics.get('macro_f1')}")

    print("\n--- Baselines (temporal test) ---")
    for name, model in [("Majority class", DummyClassifier(strategy="most_frequent")),
                        ("Trend rule", RuleBaseline()),
                        ("Logistic Regression", make_lr())]:
        fit_model(model, Xtr, ytr)
        add(name, "temporal_test", evaluate(yte, model.predict(Xte), model.predict_proba(Xte)))

    print("\n--- Tuning (forward-chaining CV on the training period) ---")
    splits = temporal_cv_splits(train)
    sw = compute_sample_weight("balanced", ytr)

    rf_search = RandomizedSearchCV(make_rf(), RF_GRID, n_iter=20, scoring="f1_macro", cv=splits,
                                   random_state=SEED, n_jobs=-1)
    rf_search.fit(Xtr, ytr)
    print(f"  RF  best CV macro-F1: {rf_search.best_score_:.4f}  {rf_search.best_params_}")

    xgb_search = RandomizedSearchCV(make_xgb(), XGB_GRID, n_iter=25, scoring="f1_macro", cv=splits,
                                    random_state=SEED, n_jobs=-1)
    xgb_search.fit(Xtr, ytr, sample_weight=sw)
    print(f"  XGB best CV macro-F1: {xgb_search.best_score_:.4f}  {xgb_search.best_params_}")

    candidates = {
        "Random Forest (tuned)": (rf_search.best_estimator_, rf_search),
        "XGBoost (tuned)": (xgb_search.best_estimator_, xgb_search),
    }

    print("\n--- Tuned models (temporal test) ---")
    preds, probas = {}, {}
    for name, (model, _) in candidates.items():
        preds[name] = model.predict(Xte)
        probas[name] = model.predict_proba(Xte)
        add(name, "temporal_test", evaluate(yte, preds[name], probas[name]))

    # choose by CV score on TRAIN period (never by the test score)
    best_name = max(candidates, key=lambda n: candidates[n][1].best_score_)
    best_model = candidates[best_name][0]
    print(f"\nSelected by CV score: {best_name}")

    print("\n--- Protocol B: 5-fold GroupKFold by repository (unseen repos) ---")
    gkf = GroupKFold(n_splits=5)
    for name, est in [("Logistic Regression", make_lr()),
                      ("Random Forest (tuned)", clone(rf_search.best_estimator_)),
                      ("XGBoost (tuned)", clone(xgb_search.best_estimator_))]:
        scores = []
        for tr, va in gkf.split(df[feats], df["y"], groups=df["repository"]):
            m = fit_model(clone(est), df.iloc[tr][feats], df["y"].iloc[tr].to_numpy())
            scores.append(f1_score(df["y"].iloc[va], m.predict(df.iloc[va][feats]),
                                   average="macro", labels=[0, 1, 2], zero_division=0))
        add(name, "grouped_5fold", {"macro_f1": round(float(np.mean(scores)), 4),
                                    "macro_f1_std": round(float(np.std(scores)), 4)})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(TABLES, "model_comparison.csv"), index=False)
    print("\n", results.to_string(index=False))

    # artifacts for the report, SHAP, evaluation and the backend
    pred = preds[best_name]
    proba = probas[best_name]
    out = test[["repository", "prediction_month", "label", "y"]].copy()
    out["pred"] = pred
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = proba[:, i]
    out.to_csv(os.path.join(TABLES, "test_predictions.csv"), index=False)

    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=CLASSES,
                                            normalize="true", cmap="Blues")
    plt.title(f"{best_name}: normalised confusion matrix (temporal test)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "confusion_matrix_best.png"), dpi=200)
    plt.close()

    pd.DataFrame({"feature": feats, "importance": best_model.feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(os.path.join(TABLES, "feature_importance.csv"), index=False)

    joblib.dump({"model": best_model, "features": feats, "classes": CLASSES,
                 "model_name": best_name,
                 "params": candidates[best_name][1].best_params_},
                os.path.join(ARTIFACTS, "model.joblib"))

    test_row = results[(results["model"] == best_name) & (results["protocol"] == "temporal_test")].iloc[0]
    cv_row = results[(results["model"] == best_name) & (results["protocol"] == "grouped_5fold")].iloc[0]
    metrics = {
        "model_name": best_name,
        "train_windows": len(train), "test_windows": len(test),
        "test_metrics": {k: float(test_row[k]) for k in
                         ["macro_f1", "balanced_acc", "accuracy", "f1_abandoned", "f1_at_risk"]
                         if k in test_row},
        "grouped_cv_macro_f1": [float(cv_row["macro_f1"]), float(cv_row["macro_f1_std"])],
        "classes": CLASSES, "n_features": len(feats),
    }
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved model + tables + figures. Best: {best_name}")


if __name__ == "__main__":
    main()
