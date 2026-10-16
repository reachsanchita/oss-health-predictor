"""
Phase 11 - LSTM on 6-month sequences of the 7 monthly metrics.

Run:  python -m ml.models.train_lstm
(needs: pip install tensorflow  - works on Python 3.12)

Uses the SAME temporal split as the tree models, so results are comparable.
Appends/updates the 'LSTM' row in reports/tables/model_comparison.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from ml.models.common import (load_features, temporal_split, evaluate,
                              CLASSES, ARTIFACTS, TABLES, SEED)
from ml.features.build_features import METRICS, WINDOW

COLS = list(METRICS.values())


def build_sequences(df):
    from ml.labeling import diagnose_thresholds as dt

    raw = dt.load_data()
    series = {}
    for repo_id, repo_df in raw.groupby("repository"):
        s = dt.build_contiguous_monthly_series(repo_id, repo_df)
        series[repo_id] = (s[COLS].to_numpy(dtype=float), {ym: i for i, ym in enumerate(s["year_month"])})

    X = np.zeros((len(df), WINDOW, len(COLS)), dtype="float32")
    for k, r in enumerate(df.itertuples()):
        arr, pos = series[r.repository]
        p = pos[r.prediction_month]
        X[k] = np.log1p(arr[p - WINDOW + 1: p + 1])
    return X


def main():
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    df, _ = load_features()
    train, test = temporal_split(df)

    # validation = newest 15% of the training months, with a 3-month gap before it
    cut = np.quantile(train["month_ord"], 0.85)
    fit_df = train[train["month_ord"] < cut - 3]
    val_df = train[train["month_ord"] >= cut]

    Xf, Xv, Xt = build_sequences(fit_df), build_sequences(val_df), build_sequences(test)
    mu, sd = Xf.mean(axis=(0, 1)), Xf.std(axis=(0, 1)) + 1e-6      # train statistics only
    Xf, Xv, Xt = [(a - mu) / sd for a in (Xf, Xv, Xt)]
    yf, yv, yt = fit_df["y"].to_numpy(), val_df["y"].to_numpy(), test["y"].to_numpy()

    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=yf)
    model = keras.Sequential([
        keras.layers.Input((WINDOW, len(COLS))),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xf, yf, validation_data=(Xv, yv), epochs=60, batch_size=64,
              class_weight={i: float(w) for i, w in enumerate(cw)},
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                        restore_best_weights=True)],
              verbose=2)

    proba = model.predict(Xt, verbose=0)
    metrics = evaluate(yt, proba.argmax(axis=1), proba)
    print("\nLSTM temporal-test metrics:", metrics)

    os.makedirs(ARTIFACTS, exist_ok=True)
    model.save(os.path.join(ARTIFACTS, "lstm.keras"))

    path = os.path.join(TABLES, "model_comparison.csv")
    row = pd.DataFrame([{"model": "LSTM", "protocol": "temporal_test", **metrics}])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["model"] != "LSTM"]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(path, index=False)
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
