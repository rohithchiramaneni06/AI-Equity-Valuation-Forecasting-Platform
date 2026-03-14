"""
database_manager.py
--------------------
Handles all SQLite database operations:
- Creating tables
- Inserting and retrieving financial data
- Storing valuation and prediction results
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "stock_platform.db"


def get_connection():
    """Returns a SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Creates all required tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS companies (
            company_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker_symbol TEXT UNIQUE NOT NULL,
            company_name  TEXT,
            sector        TEXT,
            industry      TEXT,
            market_cap    REAL
        );

        CREATE TABLE IF NOT EXISTS financial_statements (
            statement_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id       INTEGER,
            year             INTEGER,
            revenue          REAL,
            operating_income REAL,
            net_income       REAL,
            total_assets     REAL,
            total_liabilities REAL,
            cash_flow        REAL,
            FOREIGN KEY (company_id) REFERENCES companies(company_id)
        );

        CREATE TABLE IF NOT EXISTS stock_prices (
            price_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id  INTEGER,
            date        TEXT,
            open_price  REAL,
            close_price REAL,
            high_price  REAL,
            low_price   REAL,
            volume      REAL,
            FOREIGN KEY (company_id) REFERENCES companies(company_id)
        );

        CREATE TABLE IF NOT EXISTS valuation_results (
            valuation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id      INTEGER,
            intrinsic_value REAL,
            terminal_value  REAL,
            discount_rate   REAL,
            margin_of_safety REAL,
            valuation_signal TEXT,
            FOREIGN KEY (company_id) REFERENCES companies(company_id)
        );

        CREATE TABLE IF NOT EXISTS prediction_results (
            prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id       INTEGER,
            prediction_date  TEXT,
            predicted_price  REAL,
            model_used       TEXT,
            confidence_score REAL,
            FOREIGN KEY (company_id) REFERENCES companies(company_id)
        );

        CREATE TABLE IF NOT EXISTS fundamental_analysis (
            analysis_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id       INTEGER,
            roe              REAL,
            debt_to_equity   REAL,
            current_ratio    REAL,
            profit_margin    REAL,
            analysis_summary TEXT,
            FOREIGN KEY (company_id) REFERENCES companies(company_id)
        );
    """)

    conn.commit()
    conn.close()


def upsert_company(ticker, name, sector, industry, market_cap):
    """Inserts or updates a company record. Returns company_id."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO companies (ticker_symbol, company_name, sector, industry, market_cap)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker_symbol) DO UPDATE SET
            company_name = excluded.company_name,
            sector       = excluded.sector,
            industry     = excluded.industry,
            market_cap   = excluded.market_cap
    """, (ticker, name, sector, industry, market_cap))

    conn.commit()

    cursor.execute("SELECT company_id FROM companies WHERE ticker_symbol = ?", (ticker,))
    row = cursor.fetchone()
    conn.close()
    return row["company_id"]


def insert_financial_statements(company_id, df):
    """Saves annual financial statement rows from a DataFrame."""
    conn = get_connection()
    # Remove existing records for this company before inserting fresh data
    conn.execute("DELETE FROM financial_statements WHERE company_id = ?", (company_id,))
    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO financial_statements
                (company_id, year, revenue, operating_income, net_income,
                 total_assets, total_liabilities, cash_flow)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            company_id,
            int(row.get("year", 0)),
            row.get("revenue"),
            row.get("operating_income"),
            row.get("net_income"),
            row.get("total_assets"),
            row.get("total_liabilities"),
            row.get("cash_flow"),
        ))
    conn.commit()
    conn.close()


def insert_stock_prices(company_id, df):
    """Saves historical stock price rows from a DataFrame."""
    conn = get_connection()
    conn.execute("DELETE FROM stock_prices WHERE company_id = ?", (company_id,))
    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO stock_prices
                (company_id, date, open_price, close_price,
                 high_price, low_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            company_id,
            str(row["date"]),
            row.get("open_price"),
            row.get("close_price"),
            row.get("high_price"),
            row.get("low_price"),
            row.get("volume"),
        ))
    conn.commit()
    conn.close()


def save_valuation_result(company_id, result: dict):
    """Saves DCF valuation output."""
    conn = get_connection()
    conn.execute("DELETE FROM valuation_results WHERE company_id = ?", (company_id,))
    conn.execute("""
        INSERT INTO valuation_results
            (company_id, intrinsic_value, terminal_value,
             discount_rate, margin_of_safety, valuation_signal)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        company_id,
        result.get("intrinsic_value"),
        result.get("terminal_value"),
        result.get("discount_rate"),
        result.get("margin_of_safety"),
        result.get("valuation_signal"),
    ))
    conn.commit()
    conn.close()


def save_prediction_result(company_id, prediction_date, predicted_price, model_used, confidence):
    """Saves a single price prediction row."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO prediction_results
            (company_id, prediction_date, predicted_price, model_used, confidence_score)
        VALUES (?, ?, ?, ?, ?)
    """, (company_id, prediction_date, predicted_price, model_used, confidence))
    conn.commit()
    conn.close()


def save_fundamental_analysis(company_id, metrics: dict, summary: str):
    """Saves fundamental analysis metrics and text summary."""
    conn = get_connection()
    conn.execute("DELETE FROM fundamental_analysis WHERE company_id = ?", (company_id,))
    conn.execute("""
        INSERT INTO fundamental_analysis
            (company_id, roe, debt_to_equity, current_ratio, profit_margin, analysis_summary)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        company_id,
        metrics.get("roe"),
        metrics.get("debt_to_equity"),
        metrics.get("current_ratio"),
        metrics.get("profit_margin"),
        summary,
    ))
    conn.commit()
    conn.close()


def get_stock_prices(company_id) -> pd.DataFrame:
    """Retrieves stored stock prices as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM stock_prices WHERE company_id = ? ORDER BY date",
        conn, params=(company_id,)
    )
    conn.close()
    return df


def get_financial_statements(company_id) -> pd.DataFrame:
    """Retrieves stored financial statements as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM financial_statements WHERE company_id = ? ORDER BY year",
        conn, params=(company_id,)
    )
    conn.close()
    return df
