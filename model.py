"""
model.py
--------
Defines, trains, and evaluates the prediction model.

Architecture: ENSEMBLE of XGBoost + Random Forest
  - Each model predicts the probability of the next-day return being positive.
  - Final probability = weighted average of both classifiers.
  - Threshold logic converts probability → BUY / SELL / HOLD signal.

Target variable:
  - 1 (UP)   if Close[t+1] > Close[t]  by more than `threshold` %
  - 0 (DOWN) if Close[t+1] < Close[t]  by more than `threshold` %
  - NaN rows where the move is too small are excluded (reduces noise).

Why classification instead of regression?
  Predicting direction (up/down) is more actionable and generally more
  accurate than trying to predict an exact future price.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ── Feature columns used by the model ──────────────────────────────────────
FEATURE_COLS = [
    "SMA_10", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26",
    "RSI",
    "ROC",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Pct",
    "ATR",
    "Vol_Ratio",
    "Daily_Return",
    "HL_Range", "Body_Size", "Upper_Wick", "Lower_Wick",
]


def build_target(df: pd.DataFrame, forward_days: int = 1,
                 threshold_pct: float = 0.3) -> pd.Series:
    """
    Create a binary classification label.

    forward_days  : How many days ahead to predict.
    threshold_pct : Minimum % move to classify as UP/DOWN (filters noise).
                    Rows where |return| < threshold are set to NaN and dropped.

    Returns:
        Series of 1 (UP) and 0 (DOWN).
    """
    future_return = df["Close"].shift(-forward_days) / df["Close"] - 1
    target = pd.Series(np.nan, index=df.index)
    target[future_return >  threshold_pct / 100] = 1   # UP
    target[future_return < -threshold_pct / 100] = 0   # DOWN
    return target


def prepare_features(df: pd.DataFrame, forward_days: int = 1,
                     threshold_pct: float = 0.3):
    """
    Build X (features) and y (labels), drop NaN rows.

    Returns:
        X         : np.ndarray  feature matrix
        y         : np.ndarray  label vector
        dates     : pd.DatetimeIndex
        scaler    : fitted StandardScaler
        feat_cols : list of feature column names used
    """
    df = df.copy()
    df["Target"] = build_target(df, forward_days, threshold_pct)

    # Only keep features that actually exist in the DataFrame
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    df.dropna(subset=feat_cols + ["Target"], inplace=True)

    X_raw = df[feat_cols].values
    y     = df["Target"].values.astype(int)
    dates = df.index

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    return X, y, dates, scaler, feat_cols


class EnsembleModel:
    """
    Weighted ensemble of XGBoost and RandomForest classifiers.

    Attributes after training:
        xgb_model  : trained XGBoostClassifier
        rf_model   : trained RandomForestClassifier
        scaler     : fitted StandardScaler (must be saved for inference)
        feat_cols  : feature columns used during training
        metrics    : dict with accuracy, AUC, classification report
    """

    def __init__(self, xgb_weight: float = 0.6):
        """
        Args:
            xgb_weight: Weight given to XGBoost vs. RandomForest (RF gets 1−xgb_weight).
        """
        self.xgb_weight = xgb_weight
        self.rf_weight  = 1 - xgb_weight

        self.xgb_model = xgb.XGBClassifier(
            n_estimators    = 300,
            max_depth       = 4,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            use_label_encoder=False,
            eval_metric     = "logloss",
            random_state    = 42,
            verbosity       = 0,
        )
        self.rf_model = RandomForestClassifier(
            n_estimators = 200,
            max_depth    = 6,
            min_samples_leaf = 5,
            random_state = 42,
            n_jobs       = -1,
        )
        self.scaler    = None
        self.feat_cols = None
        self.metrics   = {}

    # ── Training ───────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, forward_days: int = 1,
              threshold_pct: float = 0.3, test_size: float = 0.2):
        """
        Full training pipeline with time-series cross-validation.

        Args:
            df            : DataFrame with OHLCV + all technical indicators.
            forward_days  : Prediction horizon.
            threshold_pct : Noise filter (% move required to label as UP/DOWN).
            test_size     : Fraction of data held out for final evaluation.

        Returns:
            self (for chaining)
        """
        X, y, dates, scaler, feat_cols = prepare_features(df, forward_days, threshold_pct)
        self.scaler    = scaler
        self.feat_cols = feat_cols

        # ── Time-series split (no look-ahead leakage) ──────────────────
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\n[Model] Training on {split_idx} samples, testing on {len(X_test)} samples")
        print(f"[Model] Class distribution (train): UP={y_train.sum()}, DOWN={(y_train==0).sum()}")

        # ── Cross-validation on training set ───────────────────────────
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]
            xgb_tmp = xgb.XGBClassifier(**{
                **self.xgb_model.get_params(), "verbosity": 0, "use_label_encoder": False
            })
            xgb_tmp.fit(Xtr, ytr, verbose=False)
            preds = xgb_tmp.predict(Xval)
            cv_scores.append(accuracy_score(yval, preds))

        cv_mean = np.mean(cv_scores)
        cv_std  = np.std(cv_scores)
        print(f"[Model] Cross-val accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

        # ── Final training on all training data ────────────────────────
        self.xgb_model.fit(X_train, y_train, verbose=False)
        self.rf_model.fit(X_train,  y_train)

        # ── Evaluation on held-out test set ────────────────────────────
        prob_test = self._ensemble_proba(X_test)
        pred_test = (prob_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred_test)
        try:
            auc = roc_auc_score(y_test, prob_test)
        except Exception:
            auc = 0.0

        self.metrics = {
            "accuracy"   : acc,
            "auc"        : auc,
            "cv_accuracy": cv_mean,
            "cv_std"     : cv_std,
            "report"     : classification_report(y_test, pred_test,
                                                  target_names=["DOWN", "UP"]),
            "confusion"  : confusion_matrix(y_test, pred_test),
        }

        print(f"\n[Model] ── Test-Set Performance ──────────────────────────")
        print(f"         Accuracy : {acc:.3f}")
        print(f"         AUC-ROC  : {auc:.3f}")
        print(f"\n{self.metrics['report']}")

        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def _ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        """Return blended UP-probability from both models."""
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_rf  = self.rf_model.predict_proba(X)[:, 1]
        return self.xgb_weight * p_xgb + self.rf_weight * p_rf

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale raw features then return UP probability."""
        X = self.scaler.transform(X_raw)
        return self._ensemble_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """XGBoost feature importances (gain-based)."""
        imp = self.xgb_model.feature_importances_
        return pd.Series(imp, index=self.feat_cols).sort_values(ascending=False)
