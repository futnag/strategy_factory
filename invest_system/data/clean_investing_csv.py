import pandas as pd
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Investing.comの日本語ヘッダーを英語にマッピング
COLUMN_MAPPING = {
    "日付": "date",
    "終値": "close",
    "始値": "open",
    "高値": "high",
    "安値": "low",
    "出来高": "volume",
    "変化率 %": "change_pct",
}


def clean_investing_csv(
    input_path: str, output_path: str = None, keep_ohlcv: bool = True
):
    """
    Investing.comからダウンロードしたCSVをきれいにしてParquetに保存する

    Parameters
    ----------
    input_path : str
        入力CSVファイルのパス
    output_path : str, optional
        出力Parquetファイルのパス（指定しない場合はinputと同じ名前の.parquet）
    keep_ohlcv : bool
        Trueの場合、OHLCVも保持する。Falseの場合はcloseのみ
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)

    logging.info(f"Processing: {input_path.name}")

    # CSV読み込み（エンコーディング自動対応）
    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="shift_jis")

    # カラム名を英語に変換
    df = df.rename(columns=COLUMN_MAPPING)

    # 日付をdatetimeに変換（日本語形式とハイフン形式の両対応）
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 数値カラムのクリーニング
    numeric_cols = ["close", "open", "high", "low", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("K", "000", regex=False)
                .str.replace("M", "000000", regex=False)
                .replace("", pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # change_pctの処理（%を除去）
    if "change_pct" in df.columns:
        df["change_pct"] = (
            df["change_pct"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .replace("", pd.NA)
        )
        df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")

    # 必要なカラムだけ残す
    if keep_ohlcv:
        cols_to_keep = ["date", "open", "high", "low", "close", "volume", "change_pct"]
    else:
        cols_to_keep = ["date", "close"]

    df = df[[col for col in cols_to_keep if col in df.columns]]

    # 日付でソートしてインデックス化
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    # 保存
    df.to_parquet(output_path)

    logging.info(f"Saved: {output_path}")
    logging.info(f"期間: {df.index.min().date()} ～ {df.index.max().date()}")
    logging.info(f"行数: {len(df)}")

    return df


if __name__ == "__main__":
    # 使用例：単一ファイル処理
    # clean_investing_csv("./attachments/gold.txt")

    # 複数ファイルをまとめて処理したい場合は以下を編集して使用
    files_to_process = [
        "./invester_data/AUD_JPY 過去データ.csv",
        "./invester_data/EUR_JPY 過去データ.csv",
        "./invester_data/NYダウ平均株価 過去データ.csv",
        "./invester_data/S&P500 過去データ.csv",
        "./invester_data/TOPIX 過去データ.csv",
        "./invester_data/TOPIX先物 先物の過去データ.csv",
        "./invester_data/US 30 Cash 過去データ.csv",
        "./invester_data/US 500 Cash 過去データ.csv",
        "./invester_data/US Tech 100 Cash 過去データ.csv",
        "./invester_data/USD_JPY 過去データ.csv",
        "./invester_data/アルミニウム 過去データ.csv",
        "./invester_data/ナスダック総合 過去データ.csv",
        "./invester_data/ニッケル先物 過去データ.csv",
        "./invester_data/パラジウム先物 過去データ.csv",
        "./invester_data/亜鉛先物 過去データ.csv",
        "./invester_data/原油先物 WTI 過去データ.csv",
        "./invester_data/天然ガス先物 過去データ.csv",
        "./invester_data/日経225先物 先物の過去データ.csv",
        "./invester_data/日経平均株価 過去データ.csv",
        "./invester_data/白金先物 過去データ.csv",
        "./invester_data/金先物 過去データ.csv",
        "./invester_data/銀先物 過去データ.csv",
        "./invester_data/銅先物 過去データ.csv",
    ]

    for file in files_to_process:
        if os.path.exists(file):
            clean_investing_csv(file, keep_ohlcv=True)
        else:
            logging.warning(f"File not found: {file}")
