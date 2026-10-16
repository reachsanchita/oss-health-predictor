"""Run: pytest -q"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from ml.features.build_features import features_for_position, WINDOW, METRICS
from backend.app.recommend import recommend
from backend.app.security import verify_signature


def make_series(n=24, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.integers(0, 50, n) for col in METRICS.values()})
    df["year_month"] = pd.period_range("2022-01", periods=n, freq="M").astype(str)
    return df


def test_features_have_no_future_leakage():
    s = make_series()
    p = 12
    before = features_for_position(s, p)
    s2 = s.copy()
    s2.loc[p + 1:, list(METRICS.values())] = 99999          # change ONLY the future
    after = features_for_position(s2, p)
    assert before == after


def test_not_enough_history_returns_none():
    assert features_for_position(make_series(), WINDOW - 2) is None


def test_feature_values_sane():
    f = features_for_position(make_series(), 15)
    assert f["commit_zero_months6"] >= 0
    assert 0 <= f["months_since_last_commit"] <= WINDOW
    assert all(np.isfinite(v) for v in f.values())


def test_recommend_uses_negative_shap_only():
    contribs = [{"feature": "contributors_mean3", "value": 1.0, "shap": -0.4},
                {"feature": "commits_last", "value": 50.0, "shap": 0.3}]
    recs = recommend(contribs)
    assert len(recs) == 1 and "contributor" in recs[0]["title"].lower()


def test_webhook_signature():
    import hmac, hashlib
    body = b'{"a":1}'
    good = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
    assert verify_signature("s3cret", body, good)
    assert not verify_signature("s3cret", body, "sha256=bad")
    assert not verify_signature("", body, good)


def test_labeling_rules():
    from ml.labeling import diagnose_thresholds as dt
    assert dt.classify(0.9, 0) == "Healthy"
    assert dt.classify(0.5, 0) == "Declining"
    assert dt.classify(0.1, 3) == "Abandoned"
    assert dt.classify(0.1, 0) == "Declining"
    ratio, floored, capped = dt.safe_ratio(100, 0.0)
    assert ratio == dt.RATIO_CAP and floored and capped
