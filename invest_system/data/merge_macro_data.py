import pandas as pd
import os
import logging

# ==================== 設定 ====================
INPUT_DIR = "./data/supplemental"
OUTPUT_FILE = os.path.join(INPUT_DIR, "macro_extended.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 統合するファイル一覧（存在するものだけ処理）
MACRO_FILES = [
    "fred_macro.parquet",
    "us_market.parquet",
    "commodities.parquet",
    "currencies.parquet",
    "vix.parquet",
    "cpi.parquet",
]

def merge_macro_data():
    logging.info("=== Merging Macro Data Started ===")

    merged_df = None
    loaded_files = []

    for filename in MACRO_FILES:
        filepath = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(filepath):
            logging.warning(f"File not found, skipping: {filename}")
            continue

        try:
            df = pd.read_parquet(filepath)

            # インデックスをdatetimeに変換
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            if merged_df is None:
                merged_df = df
            else:
                # カラム名が重複する場合の対応
                overlapping_cols = set(merged_df.columns) & set(df.columns)
                if overlapping_cols:
                    logging.info(f"Overlapping columns found in {filename}: {overlapping_cols}")
                    # 重複カラムは右側のものをリネーム（_dupを付与）
                    df = df.rename(columns={col: f"{col}_dup" for col in overlapping_cols})

                merged_df = pd.merge(
                    merged_df,
                    df,
                    left_index=True,
                    right_index=True,
                    how="outer"
                )

            loaded_files.append(filename)
            logging.info(f"  ✓ Merged: {filename} ({len(df)} rows)")

        except Exception as e:
            logging.error(f"  ✗ Failed to merge {filename}: {e}")

    if merged_df is not None:
        # 日付でソート
        merged_df = merged_df.sort_index()

        # 重複カラムのリネーム（最終調整）
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        # 保存
        merged_df.to_parquet(OUTPUT_FILE)
        logging.info("\n=== Merge Completed ===")
        logging.info(f"Saved: {OUTPUT_FILE}")
        logging.info(f"Total rows: {len(merged_df)}")
        logging.info(f"Total columns: {len(merged_df.columns)}")
        logging.info(f"Columns: {list(merged_df.columns)}")

        return merged_df
    else:
        logging.error("No files were merged.")
        return pd.DataFrame()

if __name__ == "__main__":
    merge_macro_data()