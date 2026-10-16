"""Live analysis: collect -> features -> predict -> explain -> recommend."""
import os
import json
import datetime

import joblib
import numpy as np
import pandas as pd
from pymongo import UpdateOne

MODEL_PATH = os.path.join("ml", "models", "artifacts", "model.joblib")
METRICS_PATH = os.path.join("ml", "models", "artifacts", "metrics.json")
_state = {}


def get_bundle():
    if "bundle" not in _state:
        from ml.explain.shap_utils import explainer_for
        _state["bundle"] = joblib.load(MODEL_PATH)
        _state["explainer"] = explainer_for(_state["bundle"]["model"])
    return _state["bundle"], _state["explainer"]


def model_info():
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def analyze(full_name):
    from ml.collector.collect_all import collect_repo
    from ml.labeling import diagnose_thresholds as dt
    from ml.features.build_features import features_for_position, METRICS
    from ml.explain.shap_utils import top_contributions
    from backend.app.recommend import recommend
    from backend.app.db import snapshots

    repo_id = full_name.replace("/", "_")
    raw, status = collect_repo(full_name, repo_id)
    if raw is None:
        raise ValueError(f"Could not collect data ({status}).")
    raw["snapshot_date"] = pd.to_datetime(raw["snapshot_date"])
    series = dt.build_contiguous_monthly_series(repo_id, raw[~raw["is_partial_month"]])
    if len(series) < 6:
        raise ValueError("Less than 6 complete months of history.")

    p = len(series) - 1
    feats = features_for_position(series, p)
    bundle, explainer = get_bundle()
    X = pd.DataFrame([feats])[bundle["features"]].replace([np.inf, -np.inf], 0).fillna(0)
    proba = bundle["model"].predict_proba(X)[0]
    classes = bundle["classes"]
    pred = int(np.argmax(proba))
    inactive = bool(feats["commits_mean6"] < 1.0)

    pred_contrib = top_contributions(explainer, X, pred)
    healthy_contrib = top_contributions(explainer, X, 0)

    cols = ["year_month"] + list(METRICS.values())
    history = json.loads(series[cols].tail(24).to_json(orient="records"))
    snapshots.bulk_write([
        UpdateOne({"repository": repo_id, "year_month": r["year_month"]}, {"$set": {**r, "repository": repo_id}},
                  upsert=True) for r in json.loads(series[cols].to_json(orient="records"))])

    return {
        "full_name": full_name,
        "status": "ready",
        "error": None,
        "prediction": "Already inactive" if inactive else classes[pred],
        "model_prediction": classes[pred],
        "already_inactive": inactive,
        "probabilities": {c: float(proba[i]) for i, c in enumerate(classes)},
        "as_of_month": str(series["year_month"].iloc[-1]),
        "horizon": "next 3 months",
        "top_factors": pred_contrib[:8],
        "recommendations": [] if inactive else recommend(healthy_contrib),
        "history": history,
        "needs_refresh": False,
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }
