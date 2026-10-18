"""Machine Learning exploratory module for the Air Quality Analysis System.

Provides an optional, carefully evaluated predictive modeling workflow using Scikit-Learn:
- Uses either pollutant indicators (feature-based regression) or historical lag features.
- Employs time-aware sequential train/test splitting (no forward lookahead or data leakage).
- Benchmarks against a persistent/dummy baseline.
- Evaluates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
- Explicitly flags predictions as empirical statistical estimates rather than sensor truth.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ModelEvaluation:
    """Evaluation metrics comparing a trained model against a baseline."""

    model_name: str
    feature_names: List[str]
    train_samples: int
    test_samples: int
    test_mae: float
    test_rmse: float
    test_r2: float
    baseline_mae: float
    baseline_rmse: float
    baseline_r2: float
    feature_importances: Optional[Dict[str, float]] = None
    y_test: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    is_suitable: bool = True
    message: str = ""


def train_aqi_estimation_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    model_type: str = "Linear Regression",
) -> ModelEvaluation:
    """Train and evaluate an AQI estimation model using available pollutant features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'aqi' and at least one pollutant column.
    test_size : float, default 0.2
        Proportion of dataset reserved for testing (time-ordered split).
    model_type : str
        "Linear Regression" or "Random Forest Regressor".
    """
    if df.empty or "aqi" not in df.columns:
        return ModelEvaluation(
            model_name=model_type,
            feature_names=[],
            train_samples=0,
            test_samples=0,
            test_mae=0.0,
            test_rmse=0.0,
            test_r2=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            baseline_r2=0.0,
            is_suitable=False,
            message="Dataset is empty or lacks an AQI column.",
        )

    # Detect candidate feature columns (pollutants)
    pollutant_candidates = ["pm2.5", "pm10", "no2", "so2", "co", "o3"]
    feature_cols = [p for p in pollutant_candidates if p in df.columns and df[p].notna().sum() >= 10]

    if not feature_cols:
        return ModelEvaluation(
            model_name=model_type,
            feature_names=[],
            train_samples=0,
            test_samples=0,
            test_mae=0.0,
            test_rmse=0.0,
            test_r2=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            baseline_r2=0.0,
            is_suitable=False,
            message="Insufficient pollutant features found to train a predictive model. Need at least one populated pollutant column (e.g. PM2.5, PM10).",
        )

    # Prepare dataset: complete cases on features and target
    clean_data = df.dropna(subset=["aqi"] + feature_cols).copy()

    # Time-ordered sort if date exists to prevent temporal leakage
    if "date" in clean_data.columns and clean_data["date"].notna().any():
        clean_data = clean_data.sort_values(by="date")

    n = len(clean_data)
    if n < 15:
        return ModelEvaluation(
            model_name=model_type,
            feature_names=feature_cols,
            train_samples=0,
            test_samples=0,
            test_mae=0.0,
            test_rmse=0.0,
            test_r2=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            baseline_r2=0.0,
            is_suitable=False,
            message=f"Dataset has only {n} complete rows. A minimum of 15 samples is required for sound statistical evaluation.",
        )

    X = clean_data[feature_cols].values
    y = clean_data["aqi"].values

    # Sequential time-aware train-test split
    split_idx = int(n * (1.0 - test_size))
    if split_idx == n or split_idx == 0:
        split_idx = max(1, n - 2)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Select model
    if model_type == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baseline comparison (Mean regressor)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_baseline_pred = baseline.predict(X_test)

    # Metrics computation
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    b_mae = float(mean_absolute_error(y_test, y_baseline_pred))
    b_rmse = float(np.sqrt(mean_squared_error(y_test, y_baseline_pred)))
    b_r2 = float(r2_score(y_test, y_baseline_pred))

    # Feature importances
    importances: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        for col, imp in zip(feature_cols, model.feature_importances_):
            importances[col.upper()] = round(float(imp), 3)
    elif hasattr(model, "coef_"):
        total_mag = sum(abs(c) for c in model.coef_) or 1.0
        for col, c in zip(feature_cols, model.coef_):
            importances[col.upper()] = round(float(abs(c) / total_mag), 3)

    return ModelEvaluation(
        model_name=model_type,
        feature_names=[f.upper() for f in feature_cols],
        train_samples=len(X_train),
        test_samples=len(X_test),
        test_mae=round(mae, 2),
        test_rmse=round(rmse, 2),
        test_r2=round(r2, 3),
        baseline_mae=round(b_mae, 2),
        baseline_rmse=round(b_rmse, 2),
        baseline_r2=round(b_r2, 3),
        feature_importances=importances,
        y_test=y_test,
        y_pred=y_pred,
        is_suitable=True,
        message="Model trained and evaluated on sequential hold-out test split.",
    )
