# EquityIQ – AI Stock Valuation & Forecasting Platform

A full end-to-end AI financial analysis platform built with Python & Streamlit.

## Features
| Module | Description |
|---|---|
| **Data Acquisition** | Fetches income statements, balance sheets, cash flows, prices via yfinance |
| **SQLite Database** | Stores all data locally in 6 relational tables |
| **DCF Valuation** | WACC estimation, FCF forecast, Terminal Value, Intrinsic Value per share |
| **Relative Valuation** | P/E, EV/EBITDA, P/B, P/S vs sector peers |
| **ML Prediction** | Linear Regression + Random Forest; 30-day forecast |
| **Fundamental Analysis** | 4-pillar health score + AI text summary |
| **Power BI Export** | 5 CSV files ready for Power BI import |

## Project Structure
```
stock_platform/
├── app.py                  ← Streamlit web app (main entry point)
├── data_fetch.py           ← Yahoo Finance data retrieval
├── data_processing.py      ← Metric calculation & feature engineering
├── dcf_model.py            ← DCF valuation engine
├── relative_valuation.py   ← Peer comparison multiples
├── price_prediction.py     ← ML models (LR + RF)
├── fundamental_analysis.py ← Health scoring + text summary
├── database_manager.py     ← SQLite CRUD layer
├── dashboard_export.py     ← Power BI CSV export
├── requirements.txt
└── data/
    ├── stock_platform.db   ← SQLite database (auto-created)
    └── powerbi_export/     ← CSV exports by ticker
```

## Setup & Run

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501

## How to Use
1. Enter a ticker (e.g. `AAPL`, `TSLA`, `INFY`) in the sidebar
2. Choose forecast horizon (5 or 10 years)
3. Click **Run Analysis**
4. Explore the 6 tabs: Price, DCF, Relative, ML, Fundamentals, Export
5. Download CSVs and import into Power BI

## Power BI Files Generated
| File | Content |
|---|---|
| `01_company_overview.csv` | Key metrics summary |
| `02_financial_statements.csv` | Revenue & profit by year |
| `03_valuation_dcf.csv` | FCF forecast |
| `03_valuation_peers.csv` | Peer multiples table |
| `04_prediction_history.csv` | Actual vs predicted prices |
| `04_prediction_future.csv` | 30-day forecast |
| `05_fundamental_scores.csv` | Pillar scores |
| `05_fundamental_ratios.csv` | Key financial ratios |

## Notes
- This platform is for **educational purposes only**. Not financial advice.
- Data accuracy depends on Yahoo Finance availability.
- First run may take 10–20 seconds as models train on 2 years of price data.
