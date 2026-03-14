"""
price_prediction.py
--------------------
Predicts future stock price trends using machine learning.

Models:
- Linear Regression  (baseline, interpretable)
- Random Forest      (captures non-linear patterns)

Features used:
- Moving averages (MA20, MA50, MA200)
- Daily returns
- Volatility
- Lag features

Output:
- Predicted next-N-day prices
- Trend direction
- Model accuracy metrics
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "ma_20", "ma_50", "ma_200",
    "return_1d", "volatility",
    "lag_1", "lag_3", "lag_5", "lag_10",
    "volume"
]

TARGET_COL = "close_price"


def run_prediction(prices_df: pd.DataFrame, forecast_days: int = 30) -> dict:
    """
    Master prediction function.
    Trains both models and returns predictions + metrics.

    Args:
        prices_df:     DataFrame with feature columns already computed
                       (output of data_processing.add_price_features)
        forecast_days: How many future days to predict

    Returns:
        dict with predictions, accuracy metrics, and trend direction
    """
    df = prices_df.copy()

    # ── Validate features ─────────────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if len(available_features) < 4:
        return {"error": "Insufficient feature data for prediction."}

    X = df[available_features].values
    y = df[TARGET_COL].values

    if len(X) < 50:
        return {"error": "Not enough historical data for prediction (need 50+ data points)."}

    # ── Train/test split (80/20, time-ordered) ────────────────────
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ── Scale features ────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Train models ──────────────────────────────────────────────
    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_sc, y_train)

    # ── Test predictions ──────────────────────────────────────────
    lr_preds = lr_model.predict(X_test_sc)
    rf_preds = rf_model.predict(X_test_sc)

    lr_metrics = _compute_metrics(y_test, lr_preds)
    rf_metrics = _compute_metrics(y_test, rf_preds)

    # ── Actual vs predicted chart data ────────────────────────────
    actual_dates  = df["date"].values[split:] if "date" in df.columns else np.arange(len(y_test))
    actual_prices = y_test

    comparison_df = pd.DataFrame({
        "date":             actual_dates,
        "actual":           actual_prices,
        "lr_predicted":     lr_preds,
        "rf_predicted":     rf_preds,
    })

    # ── Future forecast ───────────────────────────────────────────
    future_predictions = _forecast_future(
        df, available_features, scaler, lr_model, rf_model, forecast_days
    )

    # ── Best model selection ──────────────────────────────────────
    best_model = "Random Forest" if rf_metrics["r2"] > lr_metrics["r2"] else "Linear Regression"

    # ── Trend direction ───────────────────────────────────────────
    current_price = float(df[TARGET_COL].iloc[-1])
    best_future   = future_predictions["rf"] if best_model == "Random Forest" else future_predictions["lr"]
    final_predicted = float(best_future[-1]) if len(best_future) > 0 else current_price
    trend = "📈 Bullish" if final_predicted > current_price else "📉 Bearish"
    trend_pct = (final_predicted - current_price) / current_price * 100

    return {
        "lr_metrics":           lr_metrics,
        "rf_metrics":           rf_metrics,
        "best_model":           best_model,
        "comparison_df":        comparison_df,
        "future_predictions":   future_predictions,
        "current_price":        current_price,
        "predicted_final":      final_predicted,
        "trend":                trend,
        "trend_pct":            trend_pct,
        "forecast_days":        forecast_days,
        "features_used":        available_features,
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred) -> dict:
    """Computes MAE, RMSE, and R² for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def _forecast_future(df, features, scaler, lr_model, rf_model, forecast_days) -> dict:
    """
    Generates future price predictions by iteratively predicting
    one step ahead and updating lag features.
    """
    lr_future = []
    rf_future = []

    # Use the last known row as the starting point
    last_row = df[features].iloc[-1].values.copy().astype(float)

    for _ in range(forecast_days):
        row_scaled = scaler.transform([last_row])

        lr_price = float(lr_model.predict(row_scaled)[0])
        rf_price = float(rf_model.predict(row_scaled)[0])

        lr_future.append(lr_price)
        rf_future.append(rf_price)

        # Shift lag features: lag_1 = latest prediction
        feature_names = list(features)
        _update_lag_features(last_row, feature_names, rf_price)

    return {"lr": lr_future, "rf": rf_future}


def _update_lag_features(row: np.ndarray, feature_names: list, new_price: float):
    """In-place update of lag feature values in the row array."""
    lag_map = {"lag_1": 1, "lag_3": 3, "lag_5": 5, "lag_10": 10}
    for lag_name, lag_val in lag_map.items():
        if lag_name in feature_names:
            idx = feature_names.index(lag_name)
            row[idx] = new_price  # simplified: shift all lags to latest price
