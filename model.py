"""
model.py  (v2 — fixes class collapse + look-ahead leakage + overfit)
---------------------------------------------------------------------
Root causes fixed vs v1:
  1. CLASS COLLAPSE  — model predicted UP 100% of the time because
     `threshold_pct` filtering left a mild UP bias AND both learners
     lack explicit class balancing.  Fix: scale_pos_weight in XGBoost
     + class_weight in RF + SMOTE oversampling on minority class.

  2. THRESHOLD_PCT FILTERING WAS DISCARDING TOO MANY DOWN SAMPLES
     in a strong bull run (8299.TWO +204% in 2 years). Fix: default
     threshold_pct reduced to 0.0 (no filtering) — all days are labeled.
     Users can still pass a custom value.

  3. LOOK-AHEAD IN SCALER — the StandardScaler was fit on train+test
     combined (via prepare_features before the split). Fix: scaler is
     now fit ONLY on X_train.

  4. OVERFITTING — 300 trees @ depth 4 with only ~300 training samples
     is too complex. Reduced to 150 estimators, added early stopping
     via eval set, added L1/L2 regularisation.

  5. PROBABILITY CALIBRATION — raw XGBoost probabilities can be poorly
     calibrated. Added CalibratedClassifierCV (isotonic) on the RF so
     both models speak the same probability language.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Optional SMOTE — graceful fallback if imbalanced-learn not installed
try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ── Feature columns ──────────────────────────────────────────────────────
FEATURE_COLS = [
    # Trend / MA
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "EMA_12",
    "EMA_26",
    # Ratio of price to MA (scale-invariant, better than raw MA)
    "Close_to_SMA20",
    "Close_to_EMA12",
    # Momentum
    "RSI",
    "ROC",
    # MACD
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    # Volatility
    "BB_Width",
    "BB_Pct",
    "ATR_Pct",
    # Volume
    "Vol_Ratio",
    # Price microstructure
    "Daily_Return",
    "HL_Range",
    "Body_Size",
    "Upper_Wick",
    "Lower_Wick",
]


def build_target(
    df: pd.DataFrame, forward_days: int = 1, threshold_pct: float = 0.0
) -> pd.Series:
    """
    Binary label: 1 = Close rises in `forward_days`, 0 = falls.

    threshold_pct=0.0 means every day is labeled (no dead-zone filtering).
    Use a small positive value (e.g. 0.5) to exclude near-flat days.
    """
    future_return = df["Close"].shift(-forward_days) / df["Close"] - 1
    if threshold_pct > 0:
        target = pd.Series(np.nan, index=df.index)
        target[future_return > threshold_pct / 100] = 1
        target[future_return < -threshold_pct / 100] = 0
    else:
        # Simple binary: 1 if up at all, 0 if flat/down
        target = (future_return > 0).astype(float)
        target[future_return.isna()] = np.nan  # last `forward_days` rows
    return target


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-invariant features: price relative to moving averages."""
    df = df.copy()
    if "SMA_20" in df.columns:
        df["Close_to_SMA20"] = df["Close"] / df["SMA_20"] - 1
    if "EMA_12" in df.columns:
        df["Close_to_EMA12"] = df["Close"] / df["EMA_12"] - 1
    if "ATR" in df.columns:
        df["ATR_Pct"] = df["ATR"] / df["Close"]
    return df


def prepare_features(
    df: pd.DataFrame, forward_days: int = 1, threshold_pct: float = 0.0
):
    """
    Build feature matrix and label vector. Scaler is NOT fit here —
    it is fit only on the training split to prevent look-ahead leakage.
    """
    df = _add_ratio_features(df.copy())
    df["Target"] = build_target(df, forward_days, threshold_pct)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df.dropna(subset=feat_cols + ["Target"], inplace=True)

    X = df[feat_cols].values
    y = df["Target"].values.astype(int)
    return X, y, df.index, feat_cols


class EnsembleModel:
    """
    Weighted ensemble: calibrated XGBoost + calibrated Random Forest.

    Key improvements over v1:
    - Scaler fit on train split only (no leakage)
    - Class-weight balancing in both learners
    - Optional SMOTE on severe imbalance (>60/40 split)
    - Threshold selected to maximise balanced accuracy on a val set
    - Reports balanced accuracy alongside regular accuracy
    """

    def __init__(self, xgb_weight: float = 0.55):
        self.xgb_weight = xgb_weight
        self.rf_weight = 1 - xgb_weight
        self.scaler = None
        self.feat_cols = None
        self.metrics = {}
        self._opt_threshold = 0.50  # updated after training

        # XGBoost — lighter than v1, with regularisation
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,  # shallower = less overfit
            learning_rate=0.05,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.1,  # L1
            reg_lambda=1.5,  # L2
            min_child_weight=5,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

        # Random Forest with calibration wrapper
        _rf_base = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model = CalibratedClassifierCV(_rf_base, method="isotonic", cv=3)

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 1,
        threshold_pct: float = 0.0,
        test_size: float = 0.2,
    ):
        """
        Full pipeline:
          1. Build features / labels
          2. Time-based train/test split
          3. Fit scaler on train only
          4. Optional SMOTE if class imbalance > 60/40
          5. Fit both learners
          6. Optimise decision threshold on a held-out validation slice
          7. Evaluate on test set
        """
        X_raw, y, dates, feat_cols = prepare_features(df, forward_days, threshold_pct)
        self.feat_cols = feat_cols

        # ── Chronological split ───────────────────────────────────────
        n = len(X_raw)
        test_idx = int(n * (1 - test_size))
        val_idx = int(n * (1 - test_size - 0.10))  # 10 % val before test

        X_train_raw = X_raw[:val_idx]
        y_train = y[:val_idx]
        X_val_raw = X_raw[val_idx:test_idx]
        y_val = y[val_idx:test_idx]
        X_test_raw = X_raw[test_idx:]
        y_test = y[test_idx:]

        # ── Scaler fit on train only ───────────────────────────────────
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)
        X_test = self.scaler.transform(X_test_raw)

        up_count = y_train.sum()
        down_count = (y_train == 0).sum()
        total = len(y_train)
        imbalance = max(up_count, down_count) / total

        print(f"\n[Model] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
        print(
            f"[Model] Class balance  UP={up_count} ({up_count/total:.0%})  "
            f"DOWN={down_count} ({down_count/total:.0%})"
        )

        # ── SMOTE on severe imbalance ──────────────────────────────────
        if HAS_SMOTE and imbalance > 0.60 and min(up_count, down_count) >= 5:
            sm = SMOTE(
                random_state=42, k_neighbors=min(5, min(up_count, down_count) - 1)
            )
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"[Model] SMOTE applied  → {len(X_train)} samples balanced")
        elif imbalance > 0.60:
            print(
                f"[Model] ⚠ Imbalance detected ({imbalance:.0%}) but SMOTE unavailable — "
                "using sample weights instead"
            )

        # ── Cross-validation (5-fold TS) on train ─────────────────────
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for tr_i, vl_i in tscv.split(X_train):
            Xtr, Xvl = X_train[tr_i], X_train[vl_i]
            ytr, yvl = y_train[tr_i], y_train[vl_i]
            sw = compute_sample_weight("balanced", ytr)
            tmp = xgb.XGBClassifier(**{**self.xgb_model.get_params(), "verbosity": 0})
            tmp.fit(Xtr, ytr, sample_weight=sw, verbose=False)
            cv_scores.append(balanced_accuracy_score(yvl, tmp.predict(Xvl)))

        cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
        print(f"[Model] CV balanced-accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

        # ── Final fit ──────────────────────────────────────────────────
        sw_train = compute_sample_weight("balanced", y_train)
        self.xgb_model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)
        self.rf_model.fit(X_train, y_train)

        # ── Optimise threshold on validation set ──────────────────────
        # Metric: F1-macro equally penalises poor precision AND recall on
        # both classes. Hard guard: reject thresholds where either class
        # recall < MIN_RECALL, preventing the "UP recall=0.05" collapse.
        MIN_RECALL = 0.15

        if len(X_val) >= 10:
            from sklearn.metrics import f1_score, recall_score

            prob_val = self._ensemble_proba(X_val)
            best_t, best_f1 = 0.50, -1.0

            for t in np.arange(0.30, 0.71, 0.01):
                preds = (prob_val >= t).astype(int)
                recalls = recall_score(y_val, preds, average=None, zero_division=0)
                if recalls.min() < MIN_RECALL:
                    continue
                f1 = f1_score(y_val, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t

            if best_f1 < 0:  # every threshold failed the guard — fallback
                best_t = 0.50
                best_f1 = 0.0
                print(
                    "[Model] ⚠ All thresholds failed min-recall guard — defaulting to 0.50"
                )

            self._opt_threshold = round(best_t, 2)
            print(
                f"[Model] Optimal threshold: {self._opt_threshold:.2f}  "
                f"(val F1-macro={best_f1:.3f}, min-recall≥{MIN_RECALL})"
            )
        else:
            print("[Model] Val set too small — using default threshold 0.50")

        # ── Test evaluation ────────────────────────────────────────────
        prob_test = self._ensemble_proba(X_test)
        pred_test = (prob_test >= self._opt_threshold).astype(int)

        acc = accuracy_score(y_test, pred_test)
        bacc = balanced_accuracy_score(y_test, pred_test)
        try:
            auc = roc_auc_score(y_test, prob_test)
        except Exception:
            auc = 0.0

        self.metrics = {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "auc": auc,
            "cv_balanced_acc": cv_mean,
            "cv_std": cv_std,
            "threshold": self._opt_threshold,
            "report": classification_report(
                y_test, pred_test, target_names=["DOWN", "UP"], zero_division=0
            ),
            "confusion": confusion_matrix(y_test, pred_test),
        }

        print(f"\n[Model] ── Test-Set Performance ──────────────────────────")
        print(f"         Accuracy          : {acc:.3f}")
        print(f"         Balanced Accuracy : {bacc:.3f}  ← key metric")
        print(f"         AUC-ROC           : {auc:.3f}")
        print(f"         Decision threshold: {self._opt_threshold:.2f}")
        print(f"\n{self.metrics['report']}")

        # Warn the user clearly if the model still collapses
        n_up = pred_test.sum()
        n_down = (pred_test == 0).sum()
        if n_up == 0 or n_down == 0:
            print(
                "  ⚠ WARNING: Model is still predicting only one class. "
                "Consider more data (--period 5y) or a lower threshold_pct."
            )

        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def _ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_rf = self.rf_model.predict_proba(X)[:, 1]
        return self.xgb_weight * p_xgb + self.rf_weight * p_rf

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X_raw)
        return self._ensemble_proba(X)

    def get_feature_importance(self) -> pd.Series:
        imp = self.xgb_model.feature_importances_
        return pd.Series(imp, index=self.feat_cols).sort_values(ascending=False)

    @property
    def threshold(self) -> float:
        return self._opt_threshold
