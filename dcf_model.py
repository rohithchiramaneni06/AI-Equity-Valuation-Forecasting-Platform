"""
dcf_model.py
-------------
Implements a full Discounted Cash Flow (DCF) valuation model.

Steps:
1. Forecast Free Cash Flow using historical growth rate
2. Estimate WACC (Weighted Average Cost of Capital)
3. Calculate Terminal Value using Gordon Growth Model
4. Discount all cash flows back to present value
5. Compute intrinsic value per share
6. Compare with current price → Margin of Safety + Signal
"""

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────
RISK_FREE_RATE      = 0.045   # ~10-year US Treasury yield
MARKET_RETURN       = 0.10    # Historical S&P 500 average annual return
TERMINAL_GROWTH_RATE = 0.025  # Conservative long-term GDP growth rate


def run_dcf(metrics: dict, forecast_years: int = 5) -> dict:
    """
    Main DCF function. Takes processed metrics dict and forecast horizon.
    Returns a results dict with intrinsic value, margin of safety, etc.
    """
    # ── 1. Estimate base Free Cash Flow ─────────────────────────────
    base_fcf = _estimate_base_fcf(metrics)
    if base_fcf is None or np.isnan(base_fcf) or base_fcf <= 0:
        return {"error": "Insufficient free cash flow data for DCF valuation."}

    # ── 2. Estimate FCF Growth Rate ──────────────────────────────────
    fcf_growth_rate = _estimate_fcf_growth(metrics)

    # ── 3. Compute WACC ─────────────────────────────────────────────
    wacc = _compute_wacc(metrics)

    # ── 4. Forecast FCF for N years ─────────────────────────────────
    projected_fcfs = []
    fcf = base_fcf
    for year in range(1, forecast_years + 1):
        fcf = fcf * (1 + fcf_growth_rate)
        projected_fcfs.append({"year": year, "fcf": fcf})

    # ── 5. Terminal Value (Gordon Growth Model) ──────────────────────
    final_fcf = projected_fcfs[-1]["fcf"]
    terminal_value = (final_fcf * (1 + TERMINAL_GROWTH_RATE)) / (wacc - TERMINAL_GROWTH_RATE)

    # ── 6. Discount all cash flows to present value ──────────────────
    pv_fcfs = 0.0
    for item in projected_fcfs:
        pv_fcfs += item["fcf"] / ((1 + wacc) ** item["year"])

    pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)

    total_intrinsic = pv_fcfs + pv_terminal

    # ── 7. Intrinsic value per share ─────────────────────────────────
    shares = metrics.get("shares_outstanding", np.nan)
    net_debt = _compute_net_debt(metrics)

    equity_value = total_intrinsic - net_debt
    if pd.notna(shares) and shares > 0:
        intrinsic_per_share = equity_value / shares
    else:
        intrinsic_per_share = np.nan

    # ── 8. Margin of Safety ──────────────────────────────────────────
    current_price = metrics.get("current_price", np.nan)
    margin_of_safety = np.nan
    valuation_signal = "Insufficient Data"

    if pd.notna(intrinsic_per_share) and pd.notna(current_price) and current_price > 0:
        margin_of_safety = (intrinsic_per_share - current_price) / intrinsic_per_share * 100

        if margin_of_safety > 20:
            valuation_signal = "🟢 Undervalued"
        elif margin_of_safety < -20:
            valuation_signal = "🔴 Overvalued"
        else:
            valuation_signal = "🟡 Fairly Valued"

    return {
        "base_fcf":           base_fcf,
        "fcf_growth_rate":    fcf_growth_rate,
        "wacc":               wacc,
        "projected_fcfs":     projected_fcfs,
        "terminal_value":     terminal_value,
        "pv_fcfs":            pv_fcfs,
        "pv_terminal":        pv_terminal,
        "total_intrinsic":    total_intrinsic,
        "intrinsic_value":    intrinsic_per_share,
        "current_price":      current_price,
        "margin_of_safety":   margin_of_safety,
        "valuation_signal":   valuation_signal,
        "discount_rate":      wacc,
        "forecast_years":     forecast_years,
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def _estimate_base_fcf(metrics: dict) -> float:
    """Uses the most recent FCF value as base. Falls back to net income."""
    fcf = metrics.get("latest_fcf")
    if pd.notna(fcf) and fcf > 0:
        return float(fcf)

    # Fallback: use net income from financials
    fin_df = metrics.get("financials_df")
    if fin_df is not None and not fin_df.empty:
        ni = fin_df["net_income"].dropna()
        if len(ni) > 0:
            return float(ni.iloc[-1])

    return np.nan


def _estimate_fcf_growth(metrics: dict) -> float:
    """
    Estimates a reasonable FCF/revenue growth rate.
    Uses revenue CAGR capped between 3% and 25%.
    """
    cagr = metrics.get("revenue_cagr", np.nan)
    if pd.notna(cagr) and cagr > 0:
        # Apply a conservative discount to revenue CAGR for FCF projection
        growth = min(cagr * 0.8, 0.25)
        return max(growth, 0.03)
    return 0.05  # Conservative default: 5%


def _compute_wacc(metrics: dict) -> float:
    """
    Estimates WACC using CAPM for cost of equity.
    Simple capital structure assumption: 70% equity, 30% debt.
    """
    beta = metrics.get("beta", 1.0)
    if not pd.notna(beta) or beta <= 0:
        beta = 1.0

    # Cost of equity via CAPM
    cost_of_equity = RISK_FREE_RATE + beta * (MARKET_RETURN - RISK_FREE_RATE)

    # Cost of debt (approximation: risk-free + spread)
    cost_of_debt = RISK_FREE_RATE + 0.02
    tax_rate = 0.21  # US corporate tax rate

    # Weights (simplified)
    equity_weight = 0.70
    debt_weight   = 0.30

    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))

    # Clamp to a reasonable range
    return max(min(wacc, 0.20), 0.06)


def _compute_net_debt(metrics: dict) -> float:
    """Estimates net debt = total debt - cash."""
    total_debt = metrics.get("total_debt", 0) or 0
    # We don't have cash directly but use total_assets - total_liabilities as equity
    return float(total_debt)
