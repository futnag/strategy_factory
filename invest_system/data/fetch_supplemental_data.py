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
OUTPUT_DIR = "./data/supplemental"  # 保存先ディレクトリ

# FREDシリーズ（日本・グローバルで有用なもの）
FRED_SERIES = {
    "japan_10y_yield": "IRLTLT01JPM156N",  # 日本10年国債利回り
    "japan_policy_rate": "IRSTCI01JPM156N",  # 日本政策金利（目安）
    "usd_jpy": "DEXJPUS",  # USD/JPY
    "vix": "VIXCLS",  # VIX
    "us_10y_yield": "DGS10",  # 米国10年債利回り
    "us_federal_funds_rate": "FEDFUNDS",  # 米FF金利
}

# yfinanceで取得するティッカー
YF_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "vix_yf": "^VIX",
    "usd_jpy_yf": "JPY=X",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== FRED取得 ====================
def fetch_fred_data():
    logging.info("Fetching FRED data...")
    if not FRED_API_KEY:
        logging.warning("FRED_API_KEY 未設定（.env）。FRED 取得をスキップ。")
        return pd.DataFrame()
    fred = Fred(api_key=FRED_API_KEY)
    all_data = {}

    for name, series_id in FRED_SERIES.items():
        try:
            series = fred.get_series(
                series_id, observation_start=START_DATE, observation_end=END_DATE
            )
            df = series.to_frame(name=name)
            df.index.name = "date"
            all_data[name] = df
            logging.info(f"  ✓ {name} ({series_id}) - {len(df)} rows")
        except Exception as e:
            logging.error(f"  ✗ Failed to fetch {name} ({series_id}): {e}")

    # 結合して保存
    if all_data:
        combined = pd.concat(all_data.values(), axis=1)
        combined = combined.sort_index()
        filepath = os.path.join(OUTPUT_DIR, "fred_macro.parquet")
        combined.to_parquet(filepath)
        logging.info(f"Saved: {filepath}")
        return combined
    return pd.DataFrame()


# ==================== yfinance取得 ====================
def fetch_yfinance_data():
    logging.info("Fetching yfinance data...")
    all_data = {}

    for name, ticker in YF_TICKERS.items():
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                logging.warning(f"  ! No data for {name}")
                continue

            # === 堅牢な Close 列取得 ===
            if isinstance(df.columns, pd.MultiIndex):
                # MultiIndex対応
                close_data = df.xs("Close", axis=1, level=0, drop_level=True)
            else:
                close_data = df["Close"]

            # 念のため DataFrame の場合は Series に変換
            if isinstance(close_data, pd.DataFrame):
                close_series = close_data.iloc[:, 0]
            else:
                close_series = close_data

            df_clean = close_series.to_frame(name=name)
            df_clean.index.name = "date"

            all_data[name] = df_clean
            logging.info(f"  ✓ {name} ({ticker}) - {len(df_clean)} rows")

        except Exception as e:
            logging.error(f"  ✗ Failed to fetch {name}: {e}")

    if all_data:
        combined = pd.concat(all_data.values(), axis=1)
        combined = combined.sort_index()
        filepath = os.path.join(OUTPUT_DIR, "us_market.parquet")
        combined.to_parquet(filepath)
        logging.info(f"Saved: {filepath}")
        return combined

    return pd.DataFrame()


# ==================== メイン実行 ====================
if __name__ == "__main__":
    logging.info("=== Supplemental Data Acquisition Started ===")

    fred_data = fetch_fred_data()
    yf_data = fetch_yfinance_data()

    # マージ処理を安全に
    if not fred_data.empty and not yf_data.empty:
        # インデックスを確実にDatetimeIndexに揃える
        fred_data.index = pd.to_datetime(fred_data.index)
        yf_data.index = pd.to_datetime(yf_data.index)

        merged = pd.merge(
            fred_data, yf_data, left_index=True, right_index=True, how="outer"
        )
        merged = merged.sort_index()

        merged_filepath = os.path.join(OUTPUT_DIR, "macro_us_combined.parquet")
        merged.to_parquet(merged_filepath)
        logging.info(f"Combined file saved: {merged_filepath}")

    logging.info("=== Data Acquisition Completed ===")
