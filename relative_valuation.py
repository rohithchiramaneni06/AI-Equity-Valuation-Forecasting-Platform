"""
relative_valuation.py
----------------------
Performs relative valuation using market multiples:
- P/E  (Price-to-Earnings)
- EV/EBITDA
- P/B  (Price-to-Book)
- P/S  (Price-to-Sales)

Compares the target company against hardcoded sector medians
and generates a peer comparison table.
"""

import numpy as np
import pandas as pd


# ── Sector benchmark medians (approximate 2024 values) ─────────────────────
# Format: sector → { multiple: median_value }
SECTOR_BENCHMARKS = {
    "Technology": {
        "pe": 28.0, "ev_ebitda": 20.0, "pb": 7.0, "ps": 6.0
    },
    "Healthcare": {
        "pe": 22.0, "ev_ebitda": 14.0, "pb": 4.5, "ps": 3.5
    },
    "Consumer Cyclical": {
        "pe": 24.0, "ev_ebitda": 12.0, "pb": 5.0, "ps": 2.0
    },
    "Consumer Defensive": {
        "pe": 20.0, "ev_ebitda": 13.0, "pb": 5.0, "ps": 1.8
    },
    "Financials": {
        "pe": 14.0, "ev_ebitda": 11.0, "pb": 1.5, "ps": 2.5
    },
    "Energy": {
        "pe": 12.0, "ev_ebitda": 7.0,  "pb": 1.8, "ps": 1.2
    },
    "Industrials": {
        "pe": 20.0, "ev_ebitda": 13.0, "pb": 4.0, "ps": 2.0
    },
    "Communication Services": {
        "pe": 18.0, "ev_ebitda": 11.0, "pb": 3.0, "ps": 2.5
    },
    "Real Estate": {
        "pe": 35.0, "ev_ebitda": 20.0, "pb": 2.0, "ps": 5.0
    },
    "Utilities": {
        "pe": 16.0, "ev_ebitda": 10.0, "pb": 1.5, "ps": 2.0
    },
    "Basic Materials": {
        "pe": 15.0, "ev_ebitda": 9.0,  "pb": 2.0, "ps": 1.5
    },
    "Default": {
        "pe": 20.0, "ev_ebitda": 12.0, "pb": 3.0, "ps": 2.5
    },
}


def run_relative_valuation(metrics: dict, sector: str = "Default") -> dict:
    """
    Main relative valuation function.
    Returns a dict with multiples, peer medians, implied prices, and signals.
    """
    company_multiples = _extract_company_multiples(metrics)
    peer_medians      = _get_sector_medians(sector)
    implied_prices    = _compute_implied_prices(metrics, peer_medians)
    comparison_df     = _build_comparison_table(company_multiples, peer_medians, implied_prices)
    signals           = _generate_signals(company_multiples, peer_medians)

    return {
        "company_multiples": company_multiples,
        "peer_medians":      peer_medians,
        "implied_prices":    implied_prices,
        "comparison_table":  comparison_df,
        "signals":           signals,
        "sector_used":       sector if sector in SECTOR_BENCHMARKS else "Default",
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def _extract_company_multiples(metrics: dict) -> dict:
    """Pulls valuation multiples from the metrics dict."""
    return {
        "pe":       metrics.get("pe_ratio",       np.nan),
        "ev_ebitda": metrics.get("ev_ebitda",     np.nan),
        "pb":        metrics.get("pb_ratio",      np.nan),
        "ps":        metrics.get("price_to_sales", np.nan),
    }


def _get_sector_medians(sector: str) -> dict:
    """Returns benchmark medians for the given sector."""
    return SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["Default"])


def _compute_implied_prices(metrics: dict, peer_medians: dict) -> dict:
    """
    Estimates implied stock price based on each multiple
    applied to the company's own fundamentals.
    """
    current_price = metrics.get("current_price", np.nan)
    implied = {}

    # Implied from P/E
    pe = peer_medians["pe"]
    fin_df = metrics.get("financials_df")
    eps = np.nan
    if fin_df is not None and not fin_df.empty:
        net_income = fin_df["net_income"].iloc[-1]
        shares = metrics.get("shares_outstanding", np.nan)
        if pd.notna(shares) and shares > 0:
            eps = net_income / shares
    if pd.notna(eps) and eps > 0:
        implied["pe_implied"] = pe * eps
    else:
        implied["pe_implied"] = np.nan

    # Implied from EV/EBITDA
    ebitda = metrics.get("ebitda", np.nan)
    ev_ebitda_mult = peer_medians["ev_ebitda"]
    shares = metrics.get("shares_outstanding", np.nan)
    total_debt = metrics.get("total_debt", 0) or 0
    if pd.notna(ebitda) and ebitda > 0 and pd.notna(shares) and shares > 0:
        implied_ev = ebitda * ev_ebitda_mult
        implied["ev_ebitda_implied"] = max((implied_ev - total_debt) / shares, 0)
    else:
        implied["ev_ebitda_implied"] = np.nan

    # Implied from P/B
    pb_mult = peer_medians["pb"]
    pb_actual = metrics.get("pb_ratio", np.nan)
    if pd.notna(pb_actual) and pb_actual > 0 and pd.notna(current_price):
        book_per_share = current_price / pb_actual
        implied["pb_implied"] = pb_mult * book_per_share
    else:
        implied["pb_implied"] = np.nan

    # Implied from P/S
    ps_mult = peer_medians["ps"]
    ps_actual = metrics.get("price_to_sales", np.nan)
    if pd.notna(ps_actual) and ps_actual > 0 and pd.notna(current_price):
        sales_per_share = current_price / ps_actual
        implied["ps_implied"] = ps_mult * sales_per_share
    else:
        implied["ps_implied"] = np.nan

    return implied


def _build_comparison_table(company_multiples: dict, peer_medians: dict, implied_prices: dict) -> pd.DataFrame:
    """Builds a clean comparison DataFrame for display."""
    rows = []

    labels = {
        "pe":       ("P/E Ratio",    "pe_implied"),
        "ev_ebitda": ("EV/EBITDA",  "ev_ebitda_implied"),
        "pb":       ("P/B Ratio",    "pb_implied"),
        "ps":       ("P/S Ratio",    "ps_implied"),
    }

    for key, (label, implied_key) in labels.items():
        company_val = company_multiples.get(key, np.nan)
        peer_val    = peer_medians.get(key, np.nan)
        implied_val = implied_prices.get(implied_key, np.nan)

        if pd.notna(company_val) and pd.notna(peer_val):
            vs_peer = f"+{(company_val/peer_val - 1)*100:.1f}%" if company_val > peer_val else f"{(company_val/peer_val - 1)*100:.1f}%"
        else:
            vs_peer = "N/A"

        rows.append({
            "Multiple":      label,
            "Company":       round(company_val, 2) if pd.notna(company_val) else "N/A",
            "Sector Median": round(peer_val, 2) if pd.notna(peer_val) else "N/A",
            "vs Peers":      vs_peer,
            "Implied Price": f"${implied_val:.2f}" if pd.notna(implied_val) else "N/A",
        })

    return pd.DataFrame(rows)


def _generate_signals(company_multiples: dict, peer_medians: dict) -> list:
    """Generates human-readable signal strings for each multiple."""
    signals = []
    for key in ["pe", "ev_ebitda", "pb", "ps"]:
        comp = company_multiples.get(key, np.nan)
        peer = peer_medians.get(key, np.nan)
        if pd.notna(comp) and pd.notna(peer):
            ratio = comp / peer
            if ratio > 1.3:
                signals.append(f"{key.upper()}: 🔴 Trading at a premium ({ratio:.1f}x sector median)")
            elif ratio < 0.7:
                signals.append(f"{key.upper()}: 🟢 Trading at a discount ({ratio:.1f}x sector median)")
            else:
                signals.append(f"{key.upper()}: 🟡 In line with sector ({ratio:.1f}x sector median)")
    return signals
