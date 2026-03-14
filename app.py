"""
app.py
-------
Main Streamlit web application for the
AI Stock Valuation and Forecasting Platform.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Internal modules ──────────────────────────────────────────────────────────
from database_manager   import initialize_database, upsert_company, insert_financial_statements, insert_stock_prices, save_valuation_result, save_prediction_result, save_fundamental_analysis
from data_fetch         import fetch_all_data
from data_processing    import compute_derived_metrics, add_price_features
from dcf_model          import run_dcf
from relative_valuation import run_relative_valuation
from price_prediction   import run_prediction
from fundamental_analysis import run_fundamental_analysis
from dashboard_export   import export_all


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockAI – Equity Valuation Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .metric-card {
        background: #0f1117;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #7a8799; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #e8edf3; font-size: 1.6rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
    .metric-delta { font-size: 0.85rem; margin-top: 4px; }

    .signal-green  { color: #22c55e; font-weight: 600; }
    .signal-red    { color: #ef4444; font-weight: 600; }
    .signal-yellow { color: #f59e0b; font-weight: 600; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2530;
        border-radius: 6px;
        color: #7a8799;
        padding: 8px 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important;
        color: #fff !important;
    }
    div[data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2a3142;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialise DB ──────────────────────────────────────────────────────────────
initialize_database()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 StockAI")
    st.markdown("*Equity Valuation & Forecasting*")
    st.divider()

    ticker = st.text_input("🔍 Ticker Symbol", value="AAPL", max_chars=10).upper().strip()
    forecast_years = st.selectbox("📅 DCF Forecast Horizon", [3, 5, 7, 10], index=1)
    forecast_days  = st.slider("🔮 Price Prediction Days", 10, 90, 30, 5)

    st.divider()
    analyze_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown("#### About")
    st.caption(
        "This platform automates equity valuation using DCF, "
        "relative multiples, and machine learning — built as a "
        "minor project for financial analytics."
    )


# ── Main content ───────────────────────────────────────────────────────────────
st.title("AI Equity Valuation & Forecasting Platform")
st.caption("Enter a ticker symbol and click **Run Analysis** to begin.")

if not analyze_btn:
    # Landing state
    col1, col2, col3, col4 = st.columns(4)
    for col, icon, label, desc in [
        (col1, "💰", "DCF Valuation",        "Discounted cash flow model"),
        (col2, "📐", "Relative Valuation",   "P/E, EV/EBITDA, P/B, P/S multiples"),
        (col3, "🤖", "ML Prediction",        "Linear Regression + Random Forest"),
        (col4, "🏥", "Fundamental Health",   "Profitability, liquidity, leverage"),
    ]:
        with col:
            st.info(f"**{icon} {label}**\n\n{desc}")
    st.stop()


# ── Run Analysis ───────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**…"):
    raw_data = fetch_all_data(ticker)

info = raw_data.get("info", {})
if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
    # Try to check if we got any data at all
    if raw_data["financials_df"].empty and raw_data["prices_df"].empty:
        st.error(f"❌ Could not fetch data for ticker **{ticker}**. Please check the symbol and try again.")
        st.stop()

# ── Process data ───────────────────────────────────────────────────────────────
with st.spinner("Processing financial data…"):
    metrics = compute_derived_metrics(raw_data["financials_df"], info)
    metrics["financials_df"] = raw_data["financials_df"]
    metrics["prices_df"]     = raw_data["prices_df"]

    prices_with_features = add_price_features(raw_data["prices_df"]) if not raw_data["prices_df"].empty else pd.DataFrame()

# ── Run all engines ────────────────────────────────────────────────────────────
with st.spinner("Running valuation models…"):
    dcf_result       = run_dcf(metrics, forecast_years)
    sector           = info.get("sector", "Default")
    relative_result  = run_relative_valuation(metrics, sector)
    fundamental_result = run_fundamental_analysis(metrics)

with st.spinner("Training ML prediction model…"):
    prediction_result = run_prediction(prices_with_features, forecast_days) if not prices_with_features.empty else {"error": "No price data"}

# ── Save to database ───────────────────────────────────────────────────────────
company_id = upsert_company(
    ticker,
    info.get("longName", ticker),
    info.get("sector", ""),
    info.get("industry", ""),
    metrics.get("market_cap"),
)
if not raw_data["financials_df"].empty:
    insert_financial_statements(company_id, raw_data["financials_df"])
if not raw_data["prices_df"].empty:
    insert_stock_prices(company_id, raw_data["prices_df"])
if "error" not in dcf_result:
    save_valuation_result(company_id, dcf_result)
if "error" not in fundamental_result:
    save_fundamental_analysis(
        company_id,
        fundamental_result.get("metrics", {}),
        fundamental_result.get("summary", ""),
    )

# ── Company header ─────────────────────────────────────────────────────────────
company_name = info.get("longName", ticker)
st.markdown(f"## {company_name} ({ticker})")
st.caption(f"{info.get('sector', '')}  ·  {info.get('industry', '')}  ·  {info.get('exchange', '')}")

# ── KPI Row ────────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

current_price = metrics.get("current_price", 0) or 0
market_cap    = metrics.get("market_cap", 0) or 0
pe            = metrics.get("pe_ratio")
beta          = metrics.get("beta")
intrinsic     = dcf_result.get("intrinsic_value") if "error" not in dcf_result else None
health_score  = fundamental_result.get("overall_score", 0)

kpi1.metric("Current Price",  f"${current_price:,.2f}")
kpi2.metric("Market Cap",     f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
kpi3.metric("P/E Ratio",      f"{pe:.1f}x" if pe else "N/A")
kpi4.metric("Beta",           f"{beta:.2f}" if beta else "N/A")
kpi5.metric("Intrinsic Value",f"${intrinsic:,.2f}" if intrinsic else "N/A",
            delta=f"{dcf_result.get('margin_of_safety', 0):.1f}% MoS" if intrinsic else None)
kpi6.metric("Health Score",   f"{health_score:.0f}/100  {fundamental_result.get('grade', '')}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Stock Price",
    "💰 DCF Valuation",
    "📐 Relative Valuation",
    "🤖 ML Prediction",
    "📋 Financials",
    "🏥 Fundamental Health",
    "⬇️ Export",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Stock Price History
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Historical Stock Price")
    prices_df = raw_data["prices_df"]

    if prices_df.empty:
        st.warning("No historical price data available.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices_df["date"], y=prices_df["close_price"],
            mode="lines", name="Close Price",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"
        ))

        # Add moving averages if available
        if not prices_with_features.empty:
            for ma, color, label in [("ma_20", "#f59e0b", "MA20"), ("ma_50", "#22c55e", "MA50"), ("ma_200", "#ef4444", "MA200")]:
                if ma in prices_with_features.columns:
                    fig.add_trace(go.Scatter(
                        x=prices_with_features["date"], y=prices_with_features[ma],
                        mode="lines", name=label,
                        line=dict(color=color, width=1.5, dash="dot"),
                    ))

        fig.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        fig_vol = go.Figure(go.Bar(
            x=prices_df["date"], y=prices_df["volume"],
            marker_color="rgba(37,99,235,0.5)", name="Volume"
        ))
        fig_vol.update_layout(
            template="plotly_dark", height=180,
            xaxis_title="", yaxis_title="Volume",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DCF Valuation
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Discounted Cash Flow Valuation")

    if "error" in dcf_result:
        st.error(f"DCF Error: {dcf_result['error']}")
    else:
        # Valuation summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intrinsic Value/Share", f"${dcf_result['intrinsic_value']:,.2f}" if dcf_result.get('intrinsic_value') else "N/A")
        c2.metric("Current Price",         f"${dcf_result['current_price']:,.2f}" if dcf_result.get('current_price') else "N/A")
        c3.metric("Margin of Safety",      f"{dcf_result['margin_of_safety']:.1f}%" if dcf_result.get('margin_of_safety') else "N/A")
        c4.metric("Valuation Signal",      dcf_result.get("valuation_signal", "N/A"))

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### DCF Assumptions")
            params_df = pd.DataFrame({
                "Parameter": ["WACC (Discount Rate)", "FCF Growth Rate", "Terminal Growth Rate", "Forecast Years", "Base FCF"],
                "Value": [
                    f"{dcf_result['wacc']*100:.2f}%",
                    f"{dcf_result['fcf_growth_rate']*100:.2f}%",
                    "2.50%",
                    f"{dcf_result['forecast_years']} years",
                    f"${dcf_result['base_fcf']/1e9:.2f}B" if dcf_result.get("base_fcf") else "N/A"
                ]
            })
            st.dataframe(params_df, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("#### Value Breakdown")
            breakdown_df = pd.DataFrame({
                "Component": ["PV of Forecasted FCFs", "PV of Terminal Value", "Total Enterprise Value", "Intrinsic Value / Share"],
                "Amount ($)": [
                    f"${dcf_result['pv_fcfs']/1e9:.2f}B",
                    f"${dcf_result['pv_terminal']/1e9:.2f}B",
                    f"${dcf_result['total_intrinsic']/1e9:.2f}B",
                    f"${dcf_result['intrinsic_value']:,.2f}" if dcf_result.get('intrinsic_value') else "N/A"
                ]
            })
            st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

        # Projected FCF waterfall chart
        st.markdown("#### Projected Free Cash Flows")
        proj_fcfs = dcf_result.get("projected_fcfs", [])
        if proj_fcfs:
            fcf_df = pd.DataFrame(proj_fcfs)
            fig_fcf = go.Figure(go.Bar(
                x=[f"Year {r['year']}" for r in proj_fcfs],
                y=[r["fcf"] / 1e9 for r in proj_fcfs],
                marker_color="#2563eb",
                text=[f"${r['fcf']/1e9:.2f}B" for r in proj_fcfs],
                textposition="outside",
            ))
            fig_fcf.update_layout(
                template="plotly_dark", height=320,
                yaxis_title="Free Cash Flow ($ Billions)",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_fcf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Relative Valuation
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Relative Valuation")

    comp_table = relative_result.get("comparison_table")
    if comp_table is not None and not comp_table.empty:
        st.markdown(f"**Sector:** {relative_result['sector_used']}")
        st.dataframe(comp_table, hide_index=True, use_container_width=True)

        # Multiples comparison radar / bar chart
        company_mult = relative_result["company_multiples"]
        peer_medians  = relative_result["peer_medians"]

        multiples = ["P/E", "EV/EBITDA", "P/B", "P/S"]
        comp_vals = [company_mult.get("pe"), company_mult.get("ev_ebitda"), company_mult.get("pb"), company_mult.get("ps")]
        peer_vals = [peer_medians.get("pe"), peer_medians.get("ev_ebitda"), peer_medians.get("pb"), peer_medians.get("ps")]

        valid = [(m, c, p) for m, c, p in zip(multiples, comp_vals, peer_vals) if c and not np.isnan(c) and p and not np.isnan(p)]
        if valid:
            labels, cv, pv = zip(*valid)
            fig_mult = go.Figure(data=[
                go.Bar(name=ticker,          x=labels, y=cv, marker_color="#2563eb"),
                go.Bar(name="Sector Median", x=labels, y=pv, marker_color="#64748b"),
            ])
            fig_mult.update_layout(
                barmode="group", template="plotly_dark", height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_mult, use_container_width=True)

    # Signals
    signals = relative_result.get("signals", [])
    if signals:
        st.markdown("#### Valuation Signals")
        for s in signals:
            st.markdown(f"- {s}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: ML Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Stock Price Prediction")

    if "error" in prediction_result:
        st.warning(f"Prediction unavailable: {prediction_result['error']}")
    else:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model",       prediction_result["best_model"])
        c2.metric("Current Price",    f"${prediction_result['current_price']:,.2f}")
        c3.metric(f"Predicted ({forecast_days}d)", f"${prediction_result['predicted_final']:,.2f}",
                   delta=f"{prediction_result['trend_pct']:+.2f}%")
        c4.metric("Trend",            prediction_result["trend"])

        st.divider()

        # Model accuracy
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Linear Regression Accuracy")
            lr_m = prediction_result["lr_metrics"]
            acc_df = pd.DataFrame({
                "Metric": ["MAE", "RMSE", "R²", "MAPE"],
                "Value":  [f"${lr_m['mae']:.2f}", f"${lr_m['rmse']:.2f}", f"{lr_m['r2']:.4f}", f"{lr_m['mape']:.2f}%"]
            })
            st.dataframe(acc_df, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("#### Random Forest Accuracy")
            rf_m = prediction_result["rf_metrics"]
            acc_df2 = pd.DataFrame({
                "Metric": ["MAE", "RMSE", "R²", "MAPE"],
                "Value":  [f"${rf_m['mae']:.2f}", f"${rf_m['rmse']:.2f}", f"{rf_m['r2']:.4f}", f"{rf_m['mape']:.2f}%"]
            })
            st.dataframe(acc_df2, hide_index=True, use_container_width=True)

        # Actual vs Predicted chart
        comp_df = prediction_result["comparison_df"]
        if not comp_df.empty:
            st.markdown("#### Actual vs Predicted (Test Set)")
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=comp_df["date"], y=comp_df["actual"],      name="Actual",          line=dict(color="#ffffff", width=2)))
            fig_pred.add_trace(go.Scatter(x=comp_df["date"], y=comp_df["lr_predicted"], name="Linear Regression", line=dict(color="#f59e0b", width=1.5, dash="dot")))
            fig_pred.add_trace(go.Scatter(x=comp_df["date"], y=comp_df["rf_predicted"], name="Random Forest",   line=dict(color="#22c55e", width=1.5, dash="dash")))
            fig_pred.update_layout(
                template="plotly_dark", height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        # Future forecast chart
        future = prediction_result.get("future_predictions", {})
        if future:
            st.markdown(f"#### {forecast_days}-Day Future Forecast")
            days = list(range(1, forecast_days + 1))
            fig_future = go.Figure()
            fig_future.add_hline(y=prediction_result["current_price"], line_dash="dash",
                                  annotation_text="Current Price", line_color="#7a8799")
            fig_future.add_trace(go.Scatter(x=days, y=future.get("lr", []), name="Linear Regression", line=dict(color="#f59e0b")))
            fig_future.add_trace(go.Scatter(x=days, y=future.get("rf", []), name="Random Forest",     line=dict(color="#22c55e")))
            fig_future.update_layout(
                template="plotly_dark", height=340,
                xaxis_title="Days Ahead", yaxis_title="Predicted Price (USD)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_future, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Financial Statements
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Financial Statement Trends")

    fin_df = raw_data["financials_df"]
    if fin_df.empty:
        st.warning("No financial statement data available.")
    else:
        # Revenue + Net Income
        fig_fin = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Revenue", "Net Income", "Operating Income", "Free Cash Flow"),
            vertical_spacing=0.15,
        )
        color_map = {"Revenue": "#2563eb", "Net Income": "#22c55e", "Operating Income": "#f59e0b", "Free Cash Flow": "#a855f7"}

        def add_bar(fig, row, col, x, y, name, color):
            fig.add_trace(go.Bar(x=x, y=[v/1e9 for v in y], name=name, marker_color=color, showlegend=False), row=row, col=col)

        add_bar(fig_fin, 1, 1, fin_df["year"], fin_df["revenue"].fillna(0),          "Revenue",          "#2563eb")
        add_bar(fig_fin, 1, 2, fin_df["year"], fin_df["net_income"].fillna(0),       "Net Income",       "#22c55e")
        add_bar(fig_fin, 2, 1, fin_df["year"], fin_df["operating_income"].fillna(0), "Operating Income", "#f59e0b")
        add_bar(fig_fin, 2, 2, fin_df["year"], fin_df["cash_flow"].fillna(0),        "FCF",              "#a855f7")

        fig_fin.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=40, b=0))
        fig_fin.update_yaxes(title_text="$ Billions")
        st.plotly_chart(fig_fin, use_container_width=True)

        # Margin trends
        st.markdown("#### Margin Trends")
        fin_df_plot = metrics.get("financials_df", fin_df)
        if "net_margin" in fin_df_plot.columns:
            fig_margin = go.Figure()
            for col_name, color, label in [
                ("net_margin",       "#22c55e", "Net Margin"),
                ("operating_margin", "#f59e0b", "Operating Margin"),
            ]:
                if col_name in fin_df_plot.columns:
                    fig_margin.add_trace(go.Scatter(
                        x=fin_df_plot["year"],
                        y=(fin_df_plot[col_name] * 100),
                        name=label, mode="lines+markers",
                        line=dict(color=color, width=2),
                    ))
            fig_margin.update_layout(
                template="plotly_dark", height=280,
                yaxis_title="Margin (%)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_margin, use_container_width=True)

        # Raw data table
        with st.expander("📄 View Raw Financial Data"):
            st.dataframe(fin_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Fundamental Health
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Fundamental Financial Health")

    dim_scores = fundamental_result.get("dimension_scores", {})
    overall    = fundamental_result.get("overall_score", 0)
    grade      = fundamental_result.get("grade", "N/A")

    # Score gauge
    col_gauge, col_summary = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall,
            delta={"reference": 60},
            title={"text": f"Health Score<br><span style='font-size:1.2rem;color:#9ca3af'>Grade: {grade}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 35],  "color": "#1f2937"},
                    {"range": [35, 60], "color": "#374151"},
                    {"range": [60, 80], "color": "#4b5563"},
                    {"range": [80, 100],"color": "#6b7280"},
                ],
                "threshold": {"line": {"color": "#22c55e", "width": 4}, "value": 75},
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_summary:
        st.markdown("#### AI-Generated Analysis")
        st.info(fundamental_result.get("summary", "No summary available."))

    # Dimension breakdown
    st.markdown("#### Dimension Scores")
    dim_cols = st.columns(len(dim_scores))
    for col, (dim_name, dim_score) in zip(dim_cols, dim_scores.items()):
        color = "normal" if dim_score >= 60 else ("inverse" if dim_score < 40 else "off")
        col.metric(dim_name, f"{dim_score:.0f}/100")

    # Key ratios
    st.divider()
    st.markdown("#### Key Financial Ratios")
    m = fundamental_result.get("metrics", {})
    ratio_data = {
        "Metric":       ["ROE", "Net Margin", "Operating Margin", "Current Ratio", "Debt/Equity", "Asset Turnover"],
        "Value":        [
            f"{m.get('roe', 0)*100:.1f}%"        if m.get("roe") else "N/A",
            f"{m.get('net_margin', 0)*100:.1f}%"  if m.get("net_margin") else "N/A",
            f"{m.get('operating_margin', 0)*100:.1f}%" if m.get("operating_margin") else "N/A",
            f"{m.get('current_ratio', 0):.2f}x"  if m.get("current_ratio") else "N/A",
            f"{m.get('debt_to_equity', 0):.2f}x" if m.get("debt_to_equity") else "N/A",
            f"{m.get('asset_turnover', 0):.2f}x" if m.get("asset_turnover") else "N/A",
        ],
        "Assessment": [
            "Strong" if m.get("roe", 0) > 0.15 else ("Moderate" if m.get("roe", 0) > 0.05 else "Weak"),
            "Strong" if m.get("net_margin", 0) > 0.15 else ("Moderate" if m.get("net_margin", 0) > 0.05 else "Weak"),
            "Strong" if m.get("operating_margin", 0) > 0.15 else "Moderate",
            "Healthy" if m.get("current_ratio", 0) > 1.5 else "Tight",
            "Low Risk" if m.get("debt_to_equity", 99) < 0.5 else ("Moderate Risk" if m.get("debt_to_equity", 99) < 1.5 else "High Risk"),
            "Efficient" if m.get("asset_turnover", 0) > 0.8 else "Average",
        ]
    }
    st.dataframe(pd.DataFrame(ratio_data), hide_index=True, use_container_width=True)

    # Spider chart
    if dim_scores:
        categories = list(dim_scores.keys())
        values     = list(dim_scores.values())
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig_spider = go.Figure(go.Scatterpolar(
            r=values_closed, theta=categories_closed,
            fill="toself", fillcolor="rgba(37,99,235,0.25)",
            line=dict(color="#2563eb", width=2),
            name=ticker
        ))
        fig_spider.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template="plotly_dark", height=380,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_spider, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: Export
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Export Data for Power BI")
    st.markdown(
        "Click the button below to export all analysis outputs as CSV files. "
        "Import them into **Microsoft Power BI** to build interactive dashboards."
    )

    export_btn = st.button("⬇️  Export All CSVs", type="primary")
    if export_btn:
        with st.spinner("Exporting…"):
            try:
                export_all(
                    ticker, metrics, dcf_result,
                    relative_result, prediction_result, fundamental_result
                )
                st.success("✅ Files exported to the `dashboard_exports/` folder!")
                st.markdown("""
**Files created:**
- `company_overview.csv`
- `financial_statements.csv`
- `stock_prices.csv`
- `valuation_results.csv`
- `relative_valuation.csv`
- `prediction_actuals.csv`
- `future_predictions.csv`
- `fundamental_analysis.csv`
                """)
            except Exception as e:
                st.error(f"Export failed: {e}")

    st.divider()
    st.markdown("#### Power BI Setup Instructions")
    st.markdown("""
1. Open **Power BI Desktop**
2. Click **Get Data → Text/CSV**
3. Import each CSV from the `dashboard_exports/` folder
4. Create relationships between tables on the `Ticker` column
5. Build visuals: bar charts for financials, gauge for health score, line charts for price trends

**Suggested Dashboard Pages:**
- Page 1: Company Overview (KPIs + valuation signal)
- Page 2: Financial Trends (revenue, income, margins)
- Page 3: DCF Valuation (intrinsic value vs price)
- Page 4: Stock Price Prediction (forecast chart)
- Page 5: Fundamental Health Score (spider + ratios)
    """)
