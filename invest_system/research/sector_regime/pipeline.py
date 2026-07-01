"""セクターレジーム比較パイプラインのオーケストレーション。"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from .config import (
    CYCLICAL_SECTORS,
    DEFENSIVE_SECTORS,
    PipelineConfig,
    SCORE_WEIGHTS,
)
from .data_loader import generate_synthetic_sector_daily, list_sectors, load_sector_daily
from .detectors import RegimeOutput, all_method_specs, run_detector
from .evaluation import (
    best_per_sector,
    evaluate_method,
    rank_methods,
    selection_rationale,
)
from .visualization import (
    plot_method_comparison_heatmap,
    plot_regime_on_price,
    plot_score_breakdown,
)
from .weekly import build_weekly_features, feature_matrix

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def configure_logging(*, verbose: bool = False) -> None:
    """CLI / ワーカー用のログ設定（冪等）。"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
        force=True,
    )


def _init_worker_logging() -> None:
    """ProcessPoolExecutor ワーカー起動時にログを有効化（Windows spawn 対策）。"""
    configure_logging()


@dataclass
class PipelineResult:
    """パイプライン実行結果。"""
    ranking: pd.DataFrame
    best: pd.DataFrame
    outputs: dict[str, list[RegimeOutput]] = field(default_factory=dict)
    weekly: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_text: str = ""
    sector_insight: str = ""
    method_successes: int = 0
    method_failures: int = 0


@dataclass
class _SectorRunResult:
    sector: str
    results: list[tuple[RegimeOutput, dict]]
    failures: list[tuple[str, str]]


def _spec_label(spec: dict) -> str:
    """進捗表示用の手法ラベル（実行前に決定可能な名前）。"""
    fn: Callable = spec["fn"]
    kw = spec.get("kwargs", {})
    name = fn.__name__
    if name == "walk_forward_hmm":
        return f"hmm_{kw.get('n_states', 2)}"
    if name == "walk_forward_cpd":
        return f"cpd_{kw.get('algo', 'pelt')}"
    if name == "walk_forward_gmm":
        return f"gmm_{kw.get('n_components', 2)}"
    if name == "walk_forward_markov_switching":
        return f"markov_{kw.get('n_regimes', 2)}"
    if name == "walk_forward_bocpd":
        return "bocpd"
    return name


def _run_sector_methods(
    sector: str,
    weekly: pd.DataFrame,
    config: PipelineConfig,
    method_specs: list[dict],
    *,
    show_method_progress: bool = True,
    global_task_offset: int = 0,
    global_task_total: int = 0,
) -> _SectorRunResult:
    """1セクターに全手法を適用。"""
    feat = feature_matrix(weekly, sector)
    if len(feat) < config.min_weeks:
        logger.warning(
            "sector=%s スキップ: 週数 %d < 最低 %d",
            sector, len(feat), config.min_weeks,
        )
        return _SectorRunResult(sector=sector, results=[], failures=[])

    results: list[tuple[RegimeOutput, dict]] = []
    failures: list[tuple[str, str]] = []
    n_methods = len(method_specs)

    for i, spec in enumerate(method_specs, start=1):
        label = _spec_label(spec)
        global_idx = global_task_offset + i
        kw = {
            "warmup": config.warmup_weeks,
            "refit_every": config.refit_every,
            "window": config.rolling_window,
            "random_state": config.random_state,
        }
        kw.update(spec.get("kwargs", {}))

        if show_method_progress:
            if global_task_total > 0:
                logger.info(
                    "[%d/%d] sector=%s method=%s 開始",
                    global_idx, global_task_total, sector, label,
                )
            else:
                logger.info(
                    "sector=%s [%d/%d] method=%s 開始",
                    sector, i, n_methods, label,
                )

        t0 = time.perf_counter()
        try:
            out = run_detector(spec["fn"], feat, sector=sector, config_kwargs=kw)
            met = evaluate_method(
                out, weekly, feat, skip_stability=not out.method.startswith("hmm"),
            )
            results.append((out, met))
            elapsed = time.perf_counter() - t0
            if show_method_progress:
                if global_task_total > 0:
                    logger.info(
                        "[%d/%d] sector=%s method=%s 完了 (%.1fs)",
                        global_idx, global_task_total, sector, label, elapsed,
                    )
                else:
                    logger.info(
                        "sector=%s [%d/%d] method=%s 完了 (%.1fs)",
                        sector, i, n_methods, label, elapsed,
                    )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            msg = f"{type(exc).__name__}: {exc}"
            failures.append((label, msg))
            if global_task_total > 0:
                logger.warning(
                    "[%d/%d] sector=%s method=%s 失敗 (%.1fs): %s",
                    global_idx, global_task_total, sector, label, elapsed, msg,
                )
            else:
                logger.warning(
                    "sector=%s [%d/%d] method=%s 失敗 (%.1fs): %s",
                    sector, i, n_methods, label, elapsed, msg,
                )

    return _SectorRunResult(sector=sector, results=results, failures=failures)


def _apply_sector_result(
    sector_result: _SectorRunResult,
    all_results: list[tuple[RegimeOutput, dict]],
    outputs_by_sector: dict[str, list[RegimeOutput]],
) -> tuple[int, int]:
    """1セクター分の結果を集約し、(成功数, 失敗数) を返す。"""
    if sector_result.results:
        all_results.extend(sector_result.results)
        outputs_by_sector[sector_result.sector] = [r[0] for r in sector_result.results]
    return len(sector_result.results), len(sector_result.failures)


def _sector_characteristic(sector: str) -> str:
    if sector in DEFENSIVE_SECTORS:
        return "defensive"
    if sector in CYCLICAL_SECTORS:
        return "cyclical"
    return "neutral"


def _build_sector_insight(best: pd.DataFrame) -> str:
    """セクター特性とベスト手法の関係考察。"""
    if best.empty:
        return "データ不足のため考察不可。"

    lines = ["## セクター特性とベスト手法の関係\n"]
    for kind in ("defensive", "cyclical", "neutral"):
        sub = best[best["sector"].apply(_sector_characteristic) == kind]
        if sub.empty:
            continue
        top_methods = sub["method"].value_counts()
        lines.append(f"### {kind} セクター（{len(sub)}件）")
        lines.append(f"  頻出ベスト手法: {top_methods.to_dict()}")
        lines.append(f"  平均複合スコア: {sub['composite_score'].mean():.3f}\n")

    def_best = best[best["sector"].apply(_sector_characteristic) == "defensive"]["method"].mode()
    cyc_best = best[best["sector"].apply(_sector_characteristic) == "cyclical"]["method"].mode()
    if not def_best.empty and not cyc_best.empty:
        lines.append(
            f"ディフェンシブセクターでは {def_best.iloc[0]} が最頻出。"
            f"サイクリカルセクターでは {cyc_best.iloc[0]} が最頻出。"
            "ボラの持続性の違いが手法選好に影響している可能性がある。"
        )
    return "\n".join(lines)


def run_pipeline(
    data_source: Union[str, Path] = "synthetic",
    *,
    sectors: Optional[list[str]] = None,
    config: Optional[PipelineConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> PipelineResult:
    """全パイプラインを実行。

    Parameters
    ----------
    data_source : str | Path
        - "synthetic": 合成データでデモ実行
        - ファイル/ディレクトリパス
        - "project": 本リポジトリの features/
    """
    config = config or PipelineConfig()
    output_dir = Path(output_dir) if output_dir else None
    pipeline_t0 = time.perf_counter()

    if str(data_source) == "synthetic":
        daily = generate_synthetic_sector_daily(sectors=sectors, seed=config.random_state)
    else:
        daily = load_sector_daily(data_source)

    if sectors:
        daily = daily[daily["sector"].isin(sectors)]

    avail = list_sectors(daily)
    if show_progress:
        logger.info(
            "セクター数: %d, 期間: %s 〜 %s",
            len(avail), daily["date"].min(), daily["date"].max(),
        )

    weekly = build_weekly_features(daily)
    if show_progress:
        logger.info("週足行数: %d", len(weekly))

    specs = all_method_specs()
    all_results: list[tuple[RegimeOutput, dict]] = []
    outputs_by_sector: dict[str, list[RegimeOutput]] = {}
    total_successes = 0
    total_failures = 0

    sector_list = [s for s in avail if len(feature_matrix(weekly, s)) >= config.min_weeks]
    total_tasks = len(sector_list) * len(specs)
    if show_progress:
        logger.info(
            "評価対象セクター: %d（最低%d週必要）, 手法: %d, 総タスク: %d",
            len(sector_list), config.min_weeks, len(specs), total_tasks,
        )

    if n_jobs != 1:
        max_workers = None if n_jobs < 0 else n_jobs
        sectors_done = 0
        n_sectors = len(sector_list)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker_logging,
        ) as ex:
            futs = {
                ex.submit(
                    _run_sector_methods,
                    s,
                    weekly,
                    config,
                    specs,
                    show_method_progress=show_progress,
                ): s
                for s in sector_list
            }
            for fut in as_completed(futs):
                sector_result = fut.result()
                ok, ng = _apply_sector_result(
                    sector_result, all_results, outputs_by_sector,
                )
                total_successes += ok
                total_failures += ng
                sectors_done += 1
                if show_progress:
                    logger.info(
                        "進捗: %d/%d セクター完了 (%s) — 手法 %d成功/%d失敗",
                        sectors_done,
                        n_sectors,
                        sector_result.sector,
                        ok,
                        ng,
                    )
    else:
        try:
            from tqdm import tqdm
            sector_iter: Union[list[str], object] = (
                tqdm(sector_list, desc="セクター", unit="sector")
                if show_progress
                else sector_list
            )
        except ImportError:
            sector_iter = sector_list

        for sector_idx, s in enumerate(sector_iter):
            sector_result = _run_sector_methods(
                s,
                weekly,
                config,
                specs,
                show_method_progress=show_progress,
                global_task_offset=sector_idx * len(specs),
                global_task_total=total_tasks,
            )
            ok, ng = _apply_sector_result(
                sector_result, all_results, outputs_by_sector,
            )
            total_successes += ok
            total_failures += ng
            if show_progress:
                logger.info(
                    "進捗: %d/%d セクター完了 (%s) — 手法 %d成功/%d失敗",
                    sector_idx + 1,
                    len(sector_list),
                    s,
                    ok,
                    ng,
                )

    ranking = rank_methods(all_results)
    best = best_per_sector(ranking)

    summary_lines = [
        "# セクターレジーム検知 手法比較サマリー\n",
        f"評価セクター数: {len(sector_list)}",
        f"比較手法数: {len(specs)}",
        # 失敗数を成果物に永続化する（ログだけだと保存結果が「全手法完走」に見える）
        f"手法実行: {total_successes}成功 / {total_failures}失敗",
        f"スコア重み: {json.dumps(SCORE_WEIGHTS, ensure_ascii=False)}\n",
        "## セクター別ベスト手法\n",
    ]
    for _, row in best.iterrows():
        summary_lines.append(
            f"- **{row['sector']}** → `{row['method']}` "
            f"(score={row['composite_score']:.3f}): {selection_rationale(row)}"
        )
    summary_text = "\n".join(summary_lines)
    sector_insight = _build_sector_insight(best)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(output_dir / "method_ranking.csv", index=False)
        best.to_csv(output_dir / "best_per_sector.csv", index=False)
        (output_dir / "summary.md").write_text(
            summary_text + "\n\n" + sector_insight, encoding="utf-8",
        )

        plot_method_comparison_heatmap(ranking, save_path=output_dir / "heatmap.png")
        plot_score_breakdown(best, save_path=output_dir / "score_breakdown.png")

        for s in sector_list[:6]:
            if s not in outputs_by_sector:
                continue
            sub = ranking[ranking["sector"] == s]
            top = sub.iloc[0] if not sub.empty else None
            if top is None:
                continue
            method = top["method"]
            out = next((o for o in outputs_by_sector[s] if o.method == method), None)
            if out:
                plot_regime_on_price(
                    weekly, out,
                    save_path=output_dir / f"regime_{s}_{method}.png",
                )
        plt_close()

    elapsed = time.perf_counter() - pipeline_t0
    if show_progress:
        logger.info(
            "パイプライン完了 (%.1fs): 手法 %d成功/%d失敗",
            elapsed, total_successes, total_failures,
        )
        if total_failures > 0:
            logger.warning(
                "一部手法が失敗しました。上記 WARNING ログで sector/method を確認してください。",
            )

    return PipelineResult(
        ranking=ranking,
        best=best,
        outputs=outputs_by_sector,
        weekly=weekly,
        summary_text=summary_text,
        sector_insight=sector_insight,
        method_successes=total_successes,
        method_failures=total_failures,
    )


def plt_close():
    """matplotlib の figure を閉じる。"""
    import matplotlib.pyplot as plt
    plt.close("all")