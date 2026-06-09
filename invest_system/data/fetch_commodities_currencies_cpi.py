import os
import sys
import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from invest_system.config import get_env  # noqa: E402

# ==================== 設定 ====================
FRED_API_KEY = get_env("FRED_API_KEY") or ""  # .env から読込（秘密はコードに書かない）
START_DATE = "2016-06-01"
END_DATE = "2026-06-09"
OUTPUT_DIR = "./data/supplemental"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 商品先物 ====================
COMMODITY_TICKERS = {
    "wti_crude": "CL=F",
    "gold": "GC=F",
    "copper": "HG=F",
    "silver": "SI=F",
}

def fetch_commodities():
    logging.info("Fetching Commodities data...")
    all_data = {}

    for name, ticker in COMMODITY_TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if not df.empty:
                close_series = df["Close"] if not isinstance(df.columns, pd.MultiIndex) else df.xs("Close", axis=1, level=0)
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                df_clean = close_series.to_frame(name=name)
                df_clean.index.name = "date"
                all_data[name] = df_clean
                logging.info(f"  ✓ {name} ({ticker}) - {len(df_clean)} rows")
        except Exception as e:
            logging.error(f"  ✗ Failed to fetch {name}: {e}")

    if all_data:
        combined = pd.concat(all_data.values(), axis=1).sort_index()
        filepath = os.path.join(OUTPUT_DIR, "commodities.parquet")
        combined.to_parquet(filepath)
        logging.info(f"Saved: {filepath}")
        return combined
    return pd.DataFrame()

# ==================== 主要為替 ====================
CURRENCY_TICKERS = {
    "eur_jpy": "EURJPY=X",
    "aud_jpy": "AUDJPY=X",
    "gbp_jpy": "GBPJPY=X",
}

def fetch_currencies():
    logging.info("Fetching Currencies data...")
    all_data = {}

    for name, ticker in CURRENCY_TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if not df.empty:
                close_series = df["Close"] if not isinstance(df.columns, pd.MultiIndex) else df.xs("Close", axis=1, level=0)
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                df_clean = close_series.to_frame(name=name)
                df_clean.index.name = "date"
                all_data[name] = df_clean
                logging.info(f"  ✓ {name} ({ticker}) - {len(df_clean)} rows")
        except Exception as e:
            logging.error(f"  ✗ Failed to fetch {name}: {e}")

    if all_data:
        combined = pd.concat(all_data.values(), axis=1).sort_index()
        filepath = os.path.join(OUTPUT_DIR, "currencies.parquet")
        combined.to_parquet(filepath)
        logging.info(f"Saved: {filepath}")
        return combined
    return pd.DataFrame()

# ==================== VIXボラティリティ指数（追加） ====================
def fetch_vix():
    logging.info("Fetching VIX data...")
    try:
        df = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if not df.empty:
            close_series = df["Close"] if not isinstance(df.columns, pd.MultiIndex) else df.xs("Close", axis=1, level=0)
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            df_clean = close_series.to_frame(name="vix")
            df_clean.index.name = "date"
            filepath = os.path.join(OUTPUT_DIR, "vix.parquet")
            df_clean.to_parquet(filepath)
            logging.info(f"  ✓ VIX (^VIX) - {len(df_clean)} rows")
            logging.info(f"Saved: {filepath}")
            return df_clean
    except Exception as e:
        logging.error(f"  ✗ Failed to fetch VIX: {e}")
    return pd.DataFrame()

# ==================== CPI（FRED） ====================
CPI_SERIES = {
    "japan_cpi": "JPNCPIALLMINMEI",
    "us_cpi": "CPIAUCSL",
    "us_core_cpi": "CPILFESL",
}

def fetch_cpi():
    logging.info("Fetching CPI data from FRED...")
    if not FRED_API_KEY:
        logging.warning("FRED_API_KEY 未設定（.env）。CPI 取得をスキップ。")
        return pd.DataFrame()
    fred = Fred(api_key=FRED_API_KEY)
    all_data = {}

    for name, series_id in CPI_SERIES.items():
        try:
            series = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            df = series.to_frame(name=name)
            df.index.name = "date"
            all_data[name] = df
            logging.info(f"  ✓ {name} ({series_id}) - {len(df)} rows")
        except Exception as e:
            logging.error(f"  ✗ Failed to fetch {name} ({series_id}): {e}")

    if all_data:
        combined = pd.concat(all_data.values(), axis=1).sort_index()
        filepath = os.path.join(OUTPUT_DIR, "cpi.parquet")
        combined.to_parquet(filepath)
        logging.info(f"Saved: {filepath}")
        return combined
    return pd.DataFrame()

# ==================== メイン ====================
if __name__ == "__main__":
    logging.info("=== Commodity, Currency, VIX & CPI Data Acquisition Started ===")

    # fetch_commodities()
    # fetch_currencies()
    fetch_vix()
    # fetch_cpi()

    logging.info("=== Data Acquisition Completed ===")