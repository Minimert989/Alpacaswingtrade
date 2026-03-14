import numpy as np
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

log       = logging.getLogger(__name__)
MODEL_DIR = Path("models/production")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class PurgedKFold:
    """
    Purged K-Fold cross-validator for financial time series.
    Implements the algorithm from Lopez de Prado (2018):
      - Splits indices into k sequential folds
      - Purges training samples whose index falls within the test window
      - Applies an embargo period after each test fold

    Parameters
    ----------
    n_splits     : int   number of folds (default 5)
    pct_embargo  : float fraction of total samples to embargo after test
                         (default 0.01 = 1%)
    """
    def __init__(self, n_splits=5, pct_embargo=0.01):
        self.n_splits    = n_splits
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        n          = len(X)
        embargo_sz = max(1, int(n * self.pct_embargo))
        fold_size  = n // self.n_splits
        indices    = np.arange(n)

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            # last fold absorbs remainder
            test_end   = n if fold == self.n_splits - 1 else test_start + fold_size

            test_idx    = indices[test_start:test_end]
            embargo_end = min(test_end + embargo_sz, n)
            # purge: exclude test window + embargo from train
            excluded    = set(range(test_start, embargo_end))
            train_idx   = np.array([i for i in indices if i not in excluded])

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class MetaModel:
    """
    Manual soft-voting ensemble: LightGBM + XGBoost + LogisticRegression.
    Avoids sklearn VotingClassifier's metadata routing requirement (sklearn ≥ 1.4).
    sample_weight is passed directly to each estimator that supports it.
    """
    def __init__(self):
        self.estimators = [
            ("lgbm", LGBMClassifier(
                n_estimators=500, learning_rate=0.01, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
                random_state=42,
            )),
            ("xgb", XGBClassifier(
                n_estimators=300, learning_rate=0.05,
                objective="binary:logistic", eval_metric="logloss",
                verbosity=0, random_state=42, seed=42,
            )),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
        # Pipeline for LR: impute NaN then scale (LightGBM/XGB handle NaN natively)
        self._imputer = SimpleImputer(strategy="median")
        self._scaler  = StandardScaler()

    def _prep_lr(self, X, fit=False):
        """Impute + scale for LogisticRegression."""
        if fit:
            X = self._imputer.fit_transform(X)
            X = self._scaler.fit_transform(X)
        else:
            X = self._imputer.transform(X)
            X = self._scaler.transform(X)
        return X

    def train(self, X, y, sample_weight=None):
        X_lr = self._prep_lr(X, fit=True)
        for name, est in self.estimators:
            X_in = X_lr if name == "lr" else X
            if sample_weight is not None and name in ("lgbm", "xgb"):
                est.fit(X_in, y, sample_weight=sample_weight)
            else:
                est.fit(X_in, y)

    def predict_proba_positive(self, X) -> np.ndarray:
        """Average P(y=1) across all estimators (soft voting)."""
        X_lr = self._prep_lr(X, fit=False)
        probas = []
        for name, est in self.estimators:
            X_in = X_lr if name == "lr" else X
            probas.append(est.predict_proba(X_in)[:, 1])
        return np.mean(probas, axis=0)

    def save(self, path):
        joblib.dump((self.estimators, self._imputer, self._scaler), path)

    def load(self, path):
        self.estimators, self._imputer, self._scaler = joblib.load(path)


class PlattCalibrator:
    """
    Platt scaling via LogisticRegression on logit(raw_prob).
    Using logit-transformed input gives proper signal spread vs raw probs in [0,1].
    Exposes .fit() / .transform() to match IsotonicRegression API.
    """
    def __init__(self, C=1.0):
        self._lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")

    @staticmethod
    def _logit(p):
        p = np.clip(np.array(p, dtype=float).ravel(), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        self._lr.fit(self._logit(X).reshape(-1, 1), np.array(y))
        return self

    def transform(self, X):
        return self._lr.predict_proba(self._logit(X).reshape(-1, 1))[:, 1]


def calibrate(calibrator, raw_probs: np.ndarray) -> np.ndarray:
    arr = np.array(raw_probs).ravel()
    cal = calibrator.transform(arr)
    assert cal.shape == arr.shape, \
        f"Calibrator shape mismatch: {arr.shape} → {cal.shape}"
    return cal


def purged_kfold_audit(X, y, cv):
    import pandas as pd
    n           = len(X)
    test_counts = np.zeros(n, dtype=int)
    rows        = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        test_counts[test_idx] += 1
        embargo_dropped = n - len(train_idx) - len(test_idx)
        rows.append({
            "fold":            fold_i,
            "train_size":      len(train_idx),
            "test_size":       len(test_idx),
            "train_range":     f"{train_idx.min()}–{train_idx.max()}",
            "test_range":      f"{test_idx.min()}–{test_idx.max()}",
            "embargo_dropped": embargo_dropped,
        })
    audit_df = pd.DataFrame(rows)
    log.info(f"\n=== PurgedKFold Audit ===\n{audit_df.to_string(index=False)}")
    assert (test_counts == 1).all(), (
        f"Samples not covered exactly once: "
        f"{(test_counts != 1).sum()} samples violated. "
        f"unique counts: {np.unique(test_counts, return_counts=True)}"
    )
    total_embargo = audit_df["embargo_dropped"].sum()
    log.info(
        f"[AUDIT OK] {n} samples covered exactly once. "
        f"Total embargo-dropped: {total_embargo}"
    )
    return audit_df


def build_oof_probs(model: MetaModel, X, y, sw=None) -> np.ndarray:
    """
    Manual OOF loop — avoids sklearn cross_val_predict + VotingClassifier
    metadata routing issues. Trains a fresh MetaModel on each fold's train set,
    predicts on the test set, assembles full OOF probability vector.
    """
    cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
    purged_kfold_audit(X, y, cv)

    oof = np.full(len(X), np.nan)
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_sw = sw[train_idx] if sw is not None else None
        fold_model = MetaModel()
        # train() fits the imputer on the fold's train set automatically
        fold_model.train(X[train_idx], y[train_idx], sample_weight=fold_sw)
        oof[test_idx] = fold_model.predict_proba_positive(X[test_idx])
        log.info(f"  OOF fold {fold_i+1}/5 done")

    assert not np.isnan(oof).any(), "OOF contains NaN"
    return oof


def save_manifest(threshold, prod_end_idx, feature_cols,
                  config_snapshot, note=""):
    manifest = {
        "trained_at":   datetime.utcnow().isoformat(),
        "threshold":    threshold,
        "prod_end_idx": prod_end_idx,
        "feature_cols": feature_cols,
        "config":       config_snapshot,
        "note":         note,
    }
    path = MODEL_DIR / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"[MANIFEST] saved → {path}")
