"""Shared helpers: data loading, leakage-safe splits, metrics, model factories."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, balanced_accuracy_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

FEATURES_CSV = os.path.join("data", "features", "features.csv")
ARTIFACTS = os.path.join("ml", "models", "artifacts")
TABLES = os.path.join("reports", "tables")
FIGURES = os.path.join("reports", "figures")
CLASSES = ["Healthy", "Declining", "Abandoned"]
META = ["repository", "prediction_month", "label", "at_risk", "month_ord", "y"]
SEED = 42


def load_features():
    df = pd.read_csv(FEATURES_CSV)
    df["month_ord"] = df["prediction_month"].map(lambda s: pd.Period(s, "M").ordinal)
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    feats = [c for c in df.columns if c not in META]
    df[feats] = df[feats].replace([np.inf, -np.inf], 0).fillna(0)
    for d in (ARTIFACTS, TABLES, FIGURES):
        os.makedirs(d, exist_ok=True)
    return df.reset_index(drop=True), feats


def temporal_split(df, test_share=0.2, gap=3):
    """
    Train on the past, test on the newest ~20% of windows.
    `gap` months are removed between them so no training label horizon
    (prediction month + 3) reaches into the test period.
    """
    counts = df.groupby("month_ord").size().sort_index()
    cum = counts.cumsum() / counts.sum()
    test_start = int(cum[cum >= 1 - test_share].index[0])
    train = df[df["month_ord"] <= test_start - gap - 1].reset_index(drop=True)
    test = df[df["month_ord"] >= test_start].reset_index(drop=True)
    return train, test


def temporal_cv_splits(train, n_splits=4, gap=3):
    """Forward-chaining CV over months, with the same gap rule."""
    ords = train["month_ord"].to_numpy()
    y = train["y"].to_numpy()
    chunks = np.array_split(np.sort(train["month_ord"].unique()), n_splits + 1)
    splits = []
    for k in range(1, n_splits + 1):
        val_months = chunks[k]
        tr = np.flatnonzero(ords < val_months[0] - gap)
        va = np.flatnonzero(np.isin(ords, val_months))
        if len(tr) > 50 and len(va) > 20 and len(set(y[tr])) == 3:
            splits.append((tr, va))
    if not splits:
        raise RuntimeError("No valid temporal CV folds; check the dataset.")
    return splits


def evaluate(y_true, y_pred, proba=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()),
        "f1_healthy": f[0], "f1_declining": f[1], "f1_abandoned": f[2],
        "recall_abandoned": r[2],
        "f1_at_risk": f1_score(y_true != 0, y_pred != 0, zero_division=0),
    }
    if proba is not None and len(set(y_true)) == 3:
        onehot = np.eye(3)[y_true]
        out["pr_auc_macro"] = average_precision_score(onehot, proba, average="macro")
        out["pr_auc_at_risk"] = average_precision_score(y_true != 0, 1 - proba[:, 0])
    return {k: round(float(v), 4) for k, v in out.items()}


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", C=1.0))


def make_rf(seed=SEED):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  n_jobs=1, random_state=seed)


def make_xgb(seed=SEED):
    from xgboost import XGBClassifier
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                         tree_method="hist", n_jobs=1, random_state=seed)


def fit_model(model, X, y):
    """XGBoost has no class_weight option, so use balanced sample weights."""
    if model.__class__.__name__ == "XGBClassifier":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model
