"""
dashboard_export.py
--------------------
Exports all analytical outputs to CSV files for Power BI import.

Creates:
- company_overview.csv
- financial_statements.csv
- stock_prices.csv
- valuation_results.csv
- prediction_results.csv
- fundamental_analysis.csv
"""

import pandas as pd
import numpy as np
import os


EXPORT_DIR = "dashboard_exports"


def export_all(ticker: str, metrics: dict, dcf_result: dict,
               relative_result: dict, prediction_result: dict,
               fundamental_result: dict):
    """
    Master export function. Writes all analytical results to CSV files.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)

    _export_company_overview(ticker, metrics, dcf_result, fundamental_result)
    _export_financial_statements(ticker, metrics)
    _export_stock_prices(ticker, metrics)
    _export_valuation_results(ticker, dcf_result, relative_result)
    _export_prediction_results(ticker, prediction_result)
    _export_fundamental_analysis(ticker, fundamental_result)

    print(f"✅ All CSVs exported to: {os.path.abspath(EXPORT_DIR)}/")


def _export_company_overview(ticker, metrics, dcf_result, fundamental_result):
    data = {
        "Ticker":           [ticker],
        "Current Price":    [metrics.get("current_price")],
        "Market Cap":       [metrics.get("market_cap")],
        "P/E Ratio":        [metrics.get("pe_ratio")],
        "P/B Ratio":        [metrics.get("pb_ratio")],
        "EV/EBITDA":        [metrics.get("ev_ebitda")],
        "Beta":             [metrics.get("beta")],
        "Intrinsic Value":  [dcf_result.get("intrinsic_value")],
        "Margin of Safety": [dcf_result.get("margin_of_safety")],
        "Valuation Signal": [dcf_result.get("valuation_signal")],
        "Health Score":     [fundamental_result.get("overall_score")],
        "Health Grade":     [fundamental_result.get("grade")],
    }
    pd.DataFrame(data).to_csv(f"{EXPORT_DIR}/company_overview.csv", index=False)


def _export_financial_statements(ticker, metrics):
    fin_df = metrics.get("financials_df")
    if fin_df is not None and not fin_df.empty:
        df = fin_df.copy()
        df.insert(0, "Ticker", ticker)
        df.to_csv(f"{EXPORT_DIR}/financial_statements.csv", index=False)


def _export_stock_prices(ticker, metrics):
    prices = metrics.get("prices_df")
    if prices is not None and not prices.empty:
        df = prices.copy()
        df.insert(0, "Ticker", ticker)
        df.to_csv(f"{EXPORT_DIR}/stock_prices.csv", index=False)


def _export_valuation_results(ticker, dcf_result, relative_result):
    if not dcf_result or "error" in dcf_result:
        return

    # DCF results
    dcf_data = {
        "Ticker":           [ticker],
        "WACC":             [dcf_result.get("wacc")],
        "FCF Growth Rate":  [dcf_result.get("fcf_growth_rate")],
        "Terminal Value":   [dcf_result.get("terminal_value")],
        "Intrinsic Value":  [dcf_result.get("intrinsic_value")],
        "Margin of Safety": [dcf_result.get("margin_of_safety")],
        "Signal":           [dcf_result.get("valuation_signal")],
    }
    pd.DataFrame(dcf_data).to_csv(f"{EXPORT_DIR}/valuation_results.csv", index=False)

    # Relative valuation
    comp_df = relative_result.get("comparison_table")
    if comp_df is not None and not comp_df.empty:
        comp_df.insert(0, "Ticker", ticker)
        comp_df.to_csv(f"{EXPORT_DIR}/relative_valuation.csv", index=False)


def _export_prediction_results(ticker, prediction_result):
    if not prediction_result or "error" in prediction_result:
        return

    comp_df = prediction_result.get("comparison_df")
    if comp_df is not None and not comp_df.empty:
        df = comp_df.copy()
        df.insert(0, "Ticker", ticker)
        df.to_csv(f"{EXPORT_DIR}/prediction_actuals.csv", index=False)

    # Future predictions
    future = prediction_result.get("future_predictions", {})
    n = len(future.get("rf", []))
    if n > 0:
        future_df = pd.DataFrame({
            "Ticker":   [ticker] * n,
            "Day":      list(range(1, n + 1)),
            "LR_Price": future.get("lr", [np.nan] * n),
            "RF_Price": future.get("rf", [np.nan] * n),
        })
        future_df.to_csv(f"{EXPORT_DIR}/future_predictions.csv", index=False)


def _export_fundamental_analysis(ticker, fundamental_result):
    if not fundamental_result:
        return

    m = fundamental_result.get("metrics", {})
    data = {
        "Ticker":           [ticker],
        "ROE":              [m.get("roe")],
        "Net Margin":       [m.get("net_margin")],
        "Operating Margin": [m.get("operating_margin")],
        "Current Ratio":    [m.get("current_ratio")],
        "Debt to Equity":   [m.get("debt_to_equity")],
        "Asset Turnover":   [m.get("asset_turnover")],
        "Overall Score":    [fundamental_result.get("overall_score")],
        "Grade":            [fundamental_result.get("grade")],
        "Summary":          [fundamental_result.get("summary")],
    }
    pd.DataFrame(data).to_csv(f"{EXPORT_DIR}/fundamental_analysis.csv", index=False)
