"""セクターレジーム比較パイプラインのオーケストレーション。"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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


@dataclass
class PipelineResult:
    """パイプライン実行結果。"""
    ranking: pd.DataFrame
    best: pd.DataFrame
    outputs: dict[str, list[RegimeOutput]] = field(default_factory=dict)
    weekly: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_text: str = ""
    sector_insight: str = ""


def _run_sector_methods(
    sector: str,
    weekly: pd.DataFrame,
    config: PipelineConfig,
    method_specs: list[dict],
) -> tuple[str, list[tuple[RegimeOutput, dict]]]:
    """1セクターに全手法を適用。"""
    feat = feature_matrix(weekly, sector)
    if len(feat) < config.min_weeks:
        return sector, []

    results = []
    for spec in method_specs:
        kw = {
            "warmup": config.warmup_weeks,
            "refit_every": config.refit_every,
            "window": config.rolling_window,
            "random_state": config.random_state,
        }
        kw.update(spec.get("kwargs", {}))
        try:
            out = run_detector(spec["fn"], feat, sector=sector, config_kwargs=kw)
            met = evaluate_method(out, weekly, feat, skip_stability=not out.method.startswith("hmm"))
            results.append((out, met))
        except Exception:  # noqa: BLE001
            continue
    return sector, results


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

    # ディフェンシブ vs サイクリカルで手法傾向を比較
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

    # --- 1. データ読み込み
    if str(data_source) == "synthetic":
        daily = generate_synthetic_sector_daily(sectors=sectors, seed=config.random_state)
    else:
        daily = load_sector_daily(data_source)

    if sectors:
        daily = daily[daily["sector"].isin(sectors)]

    avail = list_sectors(daily)
    if show_progress:
        print(f"セクター数: {len(avail)}, 期間: {daily['date'].min()} 〜 {daily['date'].max()}")

    # --- 2. 週足特徴量
    weekly = build_weekly_features(daily)
    if show_progress:
        print(f"週足行数: {len(weekly)}")

    # --- 3. 全手法 × 全セクター
    specs = all_method_specs()
    all_results: list[tuple[RegimeOutput, dict]] = []
    outputs_by_sector: dict[str, list[RegimeOutput]] = {}

    sector_list = [s for s in avail if len(feature_matrix(weekly, s)) >= config.min_weeks]
    if show_progress:
        print(f"評価対象セクター: {len(sector_list)}（最低{config.min_weeks}週必要）")

    if n_jobs != 1:
        with ProcessPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as ex:
            futs = {
                ex.submit(_run_sector_methods, s, weekly, config, specs): s
                for s in sector_list
            }
            for fut in as_completed(futs):
                s, res = fut.result()
                if res:
                    all_results.extend(res)
                    outputs_by_sector[s] = [r[0] for r in res]
                if show_progress:
                    print(f"  完了: {s}")
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(sector_list, desc="セクター") if show_progress else sector_list
        except ImportError:
            iterator = sector_list
        for s in iterator:
            _, res = _run_sector_methods(s, weekly, config, specs)
            if res:
                all_results.extend(res)
                outputs_by_sector[s] = [r[0] for r in res]

    # --- 4. ランキング
    ranking = rank_methods(all_results)
    best = best_per_sector(ranking)

    # --- 5. サマリーテキスト
    summary_lines = [
        "# セクターレジーム検知 手法比較サマリー\n",
        f"評価セクター数: {len(sector_list)}",
        f"比較手法数: {len(specs)}",
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

    # --- 6. 出力保存
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(output_dir / "method_ranking.csv", index=False)
        best.to_csv(output_dir / "best_per_sector.csv", index=False)
        (output_dir / "summary.md").write_text(summary_text + "\n\n" + sector_insight, encoding="utf-8")

        plot_method_comparison_heatmap(ranking, save_path=output_dir / "heatmap.png")
        plot_score_breakdown(best, save_path=output_dir / "score_breakdown.png")

        for s in sector_list[:6]:
            if s not in outputs_by_sector:
                continue
            top = ranking[ranking["sector"] == s].iloc[0] if not ranking[ranking["sector"] == s].empty else None
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

    return PipelineResult(
        ranking=ranking,
        best=best,
        outputs=outputs_by_sector,
        weekly=weekly,
        summary_text=summary_text,
        sector_insight=sector_insight,
    )


def plt_close():
    """matplotlib の figure を閉じる。"""
    import matplotlib.pyplot as plt
    plt.close("all")