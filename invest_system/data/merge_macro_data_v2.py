import pandas as pd
import os
import logging

# ==================== 設定 ====================
INPUT_DIR = "./data/supplemental"
OUTPUT_FILE = os.path.join(INPUT_DIR, "macro_extended.parquet")

# 月次データとして扱うカラム（ffill対象）
MONTHLY_COLUMNS = [
    "japan_cpi",
    "us_cpi",
    "us_core_cpi",
    "japan_10y_yield",
    "japan_policy_rate",
    "us_federal_funds_rate",
]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MACRO_FILES = [
    "fred_macro.parquet",
    "us_market.parquet",
    "commodities.parquet",
    "currencies.parquet",
    "vix.parquet",
    "cpi.parquet",
]


def load_and_standardize(filepath: str) -> pd.DataFrame:
    """ファイルを読み込んで日付インデックスを標準化"""
    df = pd.read_parquet(filepath)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """複数のDataFrameをouter joinで結合（重複カラム対応）"""
    merged = None

    for i, df in enumerate(dfs):
        if merged is None:
            merged = df
        else:
            # 重複カラムのリネーム
            overlapping = set(merged.columns) & set(df.columns)
            if overlapping:
                logging.warning(f"重複カラム検出: {overlapping} → リネームします")
                df = df.rename(columns={col: f"{col}_dup{i}" for col in overlapping})

            merged = pd.merge(
                merged, df, left_index=True, right_index=True, how="outer"
            )

    return merged.sort_index()


def forward_fill_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """月次データと思われるカラムを前日値で補完"""
    for col in MONTHLY_COLUMNS:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].ffill()
            after = df[col].isna().sum()
            logging.info(f"  {col}: ffill適用（欠損 {before} → {after}）")
    return df


def check_date_alignment(df: pd.DataFrame):
    """日付の整合性と欠損状況をチェック"""
    logging.info("=== 日付整合性チェック ===")
    logging.info(f"期間: {df.index.min().date()} ～ {df.index.max().date()}")
    logging.info(f"総日数: {len(df)}")

    for col in df.columns:
        missing = df[col].isna().sum()
        missing_ratio = missing / len(df) * 100
        logging.info(f"  {col:25} | 欠損: {missing:5} ({missing_ratio:5.1f}%)")


def main():
    logging.info("=== マクロデータ統合（改善版）開始 ===")

    dataframes = []
    loaded_files = []

    for filename in MACRO_FILES:
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(filepath):
            logging.warning(f"ファイルが見つかりません: {filename}")
            continue

        try:
            df = load_and_standardize(filepath)
            dataframes.append(df)
            loaded_files.append(filename)
            logging.info(f"読み込み完了: {filename} ({len(df)}行)")
        except Exception as e:
            logging.error(f"{filename} の読み込みに失敗: {e}")

    if not dataframes:
        logging.error("マージ対象のファイルがありません。処理を終了します。")
        return

    # マージ実行
    merged_df = merge_dataframes(dataframes)

    # 月次データの補完
    merged_df = forward_fill_monthly(merged_df)

    # 日付整合性チェック
    check_date_alignment(merged_df)

    # 保存
    merged_df.to_parquet(OUTPUT_FILE)
    logging.info("\n=== 統合完了 ===")
    logging.info(f"保存先: {OUTPUT_FILE}")
    logging.info(f"最終行数: {len(merged_df)}")
    logging.info(f"カラム数: {len(merged_df.columns)}")

    # サマリー表示
    print("\n=== 統合後のカラム一覧 ===")
    for col in merged_df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
