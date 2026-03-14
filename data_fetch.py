"""
data_fetch.py
--------------
Fetches all required financial data for a given ticker symbol
using the yfinance library.

Returns structured data suitable for database storage and analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_all_data(ticker: str) -> dict:
    """
    Master function that fetches all data for a ticker.

    Returns a dict with keys:
        info, income_stmt, balance_sheet, cash_flow,
        history, financials_df, prices_df
    """
    stock = yf.Ticker(ticker)

    info = _safe_fetch_info(stock)
    income_stmt = _safe_fetch_income(stock)
    balance_sheet = _safe_fetch_balance(stock)
    cash_flow = _safe_fetch_cashflow(stock)
    history = _safe_fetch_history(stock)

    financials_df = _build_financials_df(income_stmt, balance_sheet, cash_flow)
    prices_df = _build_prices_df(history)

    return {
        "info": info,
        "income_stmt": income_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "history": history,
        "financials_df": financials_df,
        "prices_df": prices_df,
    }


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _safe_fetch_info(stock) -> dict:
    """Fetches company info safely."""
    try:
        return stock.info or {}
    except Exception:
        return {}


def _safe_fetch_income(stock) -> pd.DataFrame:
    """Fetches annual income statement."""
    try:
        df = stock.financials  # columns = dates, rows = line items
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_fetch_balance(stock) -> pd.DataFrame:
    """Fetches annual balance sheet."""
    try:
        df = stock.balance_sheet
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_fetch_cashflow(stock) -> pd.DataFrame:
    """Fetches annual cash flow statement."""
    try:
        df = stock.cashflow
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_fetch_history(stock, period="5y") -> pd.DataFrame:
    """Fetches historical OHLCV price data."""
    try:
        df = stock.history(period=period)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _get_row(df: pd.DataFrame, *possible_names) -> pd.Series:
    """
    Tries multiple possible row names and returns the first match.
    Returns a Series of NaN if nothing is found.
    """
    for name in possible_names:
        if name in df.index:
            return df.loc[name]
    # Return empty series
    if not df.empty:
        return pd.Series(np.nan, index=df.columns)
    return pd.Series(dtype=float)


def _build_financials_df(income_stmt, balance_sheet, cash_flow) -> pd.DataFrame:
    """
    Combines income statement, balance sheet, and cash flow into one
    tidy DataFrame with one row per year.
    """
    records = []

    # Determine available years from income statement columns
    if income_stmt.empty:
        return pd.DataFrame()

    for col in income_stmt.columns:
        year = pd.Timestamp(col).year

        revenue = _get_row(income_stmt, "Total Revenue", "Revenue").get(col, np.nan)
        op_income = _get_row(income_stmt, "Operating Income", "Ebit").get(col, np.nan)
        net_income = _get_row(income_stmt, "Net Income").get(col, np.nan)

        total_assets = np.nan
        total_liabilities = np.nan
        if not balance_sheet.empty:
            total_assets = _get_row(balance_sheet, "Total Assets").get(col, np.nan)
            total_liabilities = _get_row(balance_sheet, "Total Liabilities Net Minority Interest",
                                          "Total Liabilities").get(col, np.nan)

        fcf = np.nan
        if not cash_flow.empty:
            op_cf = _get_row(cash_flow, "Operating Cash Flow",
                              "Cash Flow From Continuing Operating Activities").get(col, np.nan)
            capex = _get_row(cash_flow, "Capital Expenditure",
                              "Purchase Of Property Plant And Equipment").get(col, np.nan)
            if pd.notna(op_cf) and pd.notna(capex):
                fcf = op_cf + capex  # capex is usually negative in yfinance

        records.append({
            "year": year,
            "revenue": revenue,
            "operating_income": op_income,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "cash_flow": fcf,
        })

    df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    return df


def _build_prices_df(history: pd.DataFrame) -> pd.DataFrame:
    """Converts yfinance history DataFrame into storage-ready format."""
    if history.empty:
        return pd.DataFrame()

    df = history.reset_index()
    df = df.rename(columns={
        "Date":   "date",
        "Open":   "open_price",
        "Close":  "close_price",
        "High":   "high_price",
        "Low":    "low_price",
        "Volume": "volume",
    })
    df["date"] = df["date"].astype(str).str[:10]  # Keep YYYY-MM-DD only
    return df[["date", "open_price", "close_price", "high_price", "low_price", "volume"]]
