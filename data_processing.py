"""
data_processing.py
-------------------
Cleans financial data and computes all derived analytical metrics:
- Revenue growth
- Operating margin
- Free Cash Flow
- ROE
- Debt-to-Equity
- Current Ratio
- Asset Turnover
"""

import pandas as pd
import numpy as np


def compute_derived_metrics(financials_df: pd.DataFrame, info: dict) -> dict:
    """
    Given the annual financials DataFrame and company info dict,
    computes and returns a dictionary of derived financial metrics.
    """
    metrics = {}

    if financials_df.empty:
        return metrics

    df = financials_df.copy().sort_values("year")

    # ── Revenue Growth (CAGR over available years) ──────────────────
    rev = df["revenue"].dropna()
    if len(rev) >= 2:
        start, end = rev.iloc[0], rev.iloc[-1]
        n = len(rev) - 1
        metrics["revenue_cagr"] = ((end / start) ** (1 / n) - 1) if start > 0 else np.nan
    else:
        metrics["revenue_cagr"] = np.nan

    # ── Year-over-year revenue growth list ──────────────────────────
    df["revenue_growth_yoy"] = df["revenue"].pct_change()
    metrics["revenue_growth_yoy"] = df[["year", "revenue_growth_yoy"]].dropna().to_dict("records")

    # ── Operating Margin ────────────────────────────────────────────
    df["operating_margin"] = df["operating_income"] / df["revenue"]
    metrics["avg_operating_margin"] = df["operating_margin"].mean()
    metrics["latest_operating_margin"] = df["operating_margin"].iloc[-1]

    # ── Net Profit Margin ───────────────────────────────────────────
    df["net_margin"] = df["net_income"] / df["revenue"]
    metrics["latest_net_margin"] = df["net_margin"].iloc[-1]
    metrics["avg_net_margin"] = df["net_margin"].mean()

    # ── Free Cash Flow (most recent year) ───────────────────────────
    metrics["latest_fcf"] = df["cash_flow"].iloc[-1] if not df["cash_flow"].isna().all() else np.nan
    metrics["fcf_series"] = df[["year", "cash_flow"]].dropna().to_dict("records")

    # ── ROE (Return on Equity) ──────────────────────────────────────
    # ROE = Net Income / (Total Assets - Total Liabilities)
    df["equity"] = df["total_assets"] - df["total_liabilities"]
    df["roe"] = df["net_income"] / df["equity"].replace(0, np.nan)
    metrics["latest_roe"] = df["roe"].iloc[-1]

    # ── Debt-to-Equity ──────────────────────────────────────────────
    df["debt_to_equity"] = df["total_liabilities"] / df["equity"].replace(0, np.nan)
    metrics["latest_debt_to_equity"] = df["debt_to_equity"].iloc[-1]

    # ── Asset Turnover ──────────────────────────────────────────────
    df["asset_turnover"] = df["revenue"] / df["total_assets"].replace(0, np.nan)
    metrics["latest_asset_turnover"] = df["asset_turnover"].iloc[-1]

    # ── Current Ratio (from yfinance info) ─────────────────────────
    metrics["current_ratio"] = info.get("currentRatio", np.nan)

    # ── From info dict ───────────────────────────────────────────────
    metrics["beta"]             = info.get("beta", 1.0)
    metrics["shares_outstanding"] = info.get("sharesOutstanding", np.nan)
    metrics["market_cap"]       = info.get("marketCap", np.nan)
    metrics["current_price"]    = info.get("currentPrice") or info.get("regularMarketPrice", np.nan)
    metrics["pe_ratio"]         = info.get("trailingPE", np.nan)
    metrics["pb_ratio"]         = info.get("priceToBook", np.nan)
    metrics["ev_ebitda"]        = info.get("enterpriseToEbitda", np.nan)
    metrics["price_to_sales"]   = info.get("priceToSalesTrailing12Months", np.nan)
    metrics["ebitda"]           = info.get("ebitda", np.nan)
    metrics["total_debt"]       = info.get("totalDebt", np.nan)
    metrics["enterprise_value"] = info.get("enterpriseValue", np.nan)

    # ── Processed DataFrame for charts ─────────────────────────────
    metrics["financials_df"] = df

    return metrics


def add_price_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches historical price data with ML features:
    - Moving averages (20, 50, 200 day)
    - Daily returns
    - Volatility (20-day rolling std of returns)
    - Lag features
    """
    df = prices_df.copy().sort_values("date").reset_index(drop=True)

    df["return_1d"]  = df["close_price"].pct_change()
    df["ma_20"]      = df["close_price"].rolling(20).mean()
    df["ma_50"]      = df["close_price"].rolling(50).mean()
    df["ma_200"]     = df["close_price"].rolling(200).mean()
    df["volatility"] = df["return_1d"].rolling(20).std()

    # Lag features
    for lag in [1, 3, 5, 10]:
        df[f"lag_{lag}"] = df["close_price"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df


def safe_float(value, default=np.nan):
    """Safely converts a value to float, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
