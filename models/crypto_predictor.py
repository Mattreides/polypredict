"""
polypredict/models/crypto_predictor.py

Walk-forward GradientBoosting + Platt (sigmoid) calibration.

There are no stored model files: every call trains fresh on the most recent
candles. Short-timeframe crypto regimes shift fast enough that a model trained
last week is already stale, so freshness beats persistence here.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, accuracy_score


def _make_model():
    base = GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42
    )
    # Platt scaling: sigmoid wrapper turns clustered GBM scores into usable probabilities.
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def train_and_predict(X, y, n_train: int = 500):
    """
    X, y are aligned (each row of X has a known next-bar label in y) EXCEPT the
    final live bar, whose label is still forming. Returns a dict with the live
    P(up), walk-forward CV accuracy, and Brier score.
    """
    X = X.tail(n_train + 1)
    y = y.tail(n_train + 1)

    # Last row is the live bar we predict; its label is the not-yet-known future.
    X_hist, y_hist = X.iloc[:-1], y.iloc[:-1]
    X_live = X.iloc[-1:]

    if len(X_hist) < 80:
        raise ValueError(f"Not enough history to train ({len(X_hist)} rows; need >= 80).")

    # ── walk-forward evaluation ──
    tss = TimeSeriesSplit(n_splits=5)
    accs, briers = [], []
    for tr, te in tss.split(X_hist):
        if len(np.unique(y_hist.iloc[tr])) < 2:
            continue
        m = _make_model()
        m.fit(X_hist.iloc[tr], y_hist.iloc[tr])
        p = m.predict_proba(X_hist.iloc[te])[:, 1]
        accs.append(accuracy_score(y_hist.iloc[te], (p >= 0.5).astype(int)))
        briers.append(brier_score_loss(y_hist.iloc[te], p))

    # ── final model on all history, predict the live bar ──
    final = _make_model()
    final.fit(X_hist, y_hist)
    p_up = float(final.predict_proba(X_live)[:, 1][0])

    return {
        "p_up": p_up,
        "p_down": 1 - p_up,
        "cv_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "n_train": int(len(X_hist)),
        "n_features": int(X.shape[1]),
    }
