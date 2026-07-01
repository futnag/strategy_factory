#!/usr/bin/env python3
"""セクター週足レジーム検知 — 手法比較パイプライン実行スクリプト。

使い方:
  # 合成データでデモ（データ未配置時）
  python examples/sector_regime_comparison.py

  # ローカル CSV/Parquet
  python examples/sector_regime_comparison.py --source path/to/sector_daily.parquet

  # 本リポジトリの features/（data/ 配置済みの場合）
  python examples/sector_regime_comparison.py --source project

  # 特定セクターのみ
  python examples/sector_regime_comparison.py --sectors 3300,3450,5250,7050
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.research.sector_regime import PipelineConfig, run_pipeline
from invest_system.research.sector_regime.data_loader import set_data_root
from invest_system.research.sector_regime.pipeline import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="セクターレジーム検知 手法比較")
    parser.add_argument(
        "--source", default="synthetic",
        help="データソース: synthetic | project | ファイル/ディレクトリパス",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="data/ のパス（worktree 時は main の data/ を指定。例: C:/Users/.../claude_local_sessions/data）",
    )
    parser.add_argument("--sectors", default=None, help="カンマ区切りセクターコード")
    parser.add_argument("--output", default="output/sector_regime", help="結果出力先")
    parser.add_argument("--warmup", type=int, default=104, help="ウォームアップ週数")
    parser.add_argument("--min-weeks", type=int, default=260, help="最低週数")
    parser.add_argument("--jobs", type=int, default=1, help="並列ワーカー数（1=逐次）")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG ログを有効化")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    if args.data_root:
        set_data_root(args.data_root)
        logger.info("データルート: %s", args.data_root)

    sectors = [s.strip() for s in args.sectors.split(",")] if args.sectors else None
    config = PipelineConfig(
        warmup_weeks=args.warmup,
        min_weeks=args.min_weeks,
    )

    logger.info("=" * 60)
    logger.info("セクター週足レジーム検知 — 手法比較パイプライン")
    logger.info("=" * 60)

    result = run_pipeline(
        data_source=args.source,
        sectors=sectors,
        config=config,
        output_dir=args.output,
        n_jobs=args.jobs,
    )

    print("\n--- セクター別ベスト手法 ---")
    if result.best.empty:
        print("（評価結果なし — データ期間またはセクター数を確認してください）")
    else:
        print(result.best[[
            "sector", "method", "composite_score",
            "pred_score", "quality_score", "stability_score", "econ_score",
        ]].to_string(index=False))

    print("\n--- 手法比較サマリー（上位10） ---")
    if not result.ranking.empty:
        top = result.ranking.nlargest(10, "composite_score")
        print(top[["sector", "method", "composite_score", "pred_hit_rate"]].to_string(index=False))

    print(f"\n結果保存先: {args.output}/")
    print(result.sector_insight)
    if result.method_failures > 0:
        logger.warning(
            "完了（警告あり）: 手法 %d成功 / %d失敗",
            result.method_successes,
            result.method_failures,
        )
    else:
        logger.info("完了: 手法 %d成功", result.method_successes)


if __name__ == "__main__":
    main()