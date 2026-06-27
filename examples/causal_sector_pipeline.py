"""33業種セクター因果的レジーム検知パイプライン（統合実行）。

1. セクター日次パネル構築
2. PCMCI+ 因果グラフ学習 + collider 同定
3. VALUE→RET 構造ブレイク検知（CUSUM / ruptures / グラフ差分）
4. DoWhy + EconML 因果効果推定 + refutation
5. 統計的レジーム vs 因果的レジーム比較
6. 可視化出力 → data/reports/causal_sector/

実行: .venv\\Scripts\\python.exe examples\\causal_sector_pipeline.py
      .venv\\Scripts\\python.exe examples\\causal_sector_pipeline.py --all-sectors
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invest_system.research.causal_sector.config import (  # noqa: E402
    JP_STRUCTURAL_EVENTS, REPRESENTATIVE_SECTORS, SECTOR_PROFILES,
)
from invest_system.research.causal_sector.panels import build_sector_daily_panel  # noqa: E402
from invest_system.research.causal_sector.graph import (  # noqa: E402
    graph_summary, identify_colliders, learn_pcmci_graph, plot_causal_graph,
)
from invest_system.research.causal_sector.regime import (  # noqa: E402
    compare_detection_methods, detect_structural_breaks,
)
from invest_system.research.causal_sector.effects import (  # noqa: E402
    estimate_causal_effect, run_refutation_tests,
)

REPORT_DIR = ROOT / "data" / "reports" / "causal_sector"
FINAL_REPORT = ROOT / "docs" / "32-causal-sector-final-report.md"


def run_sector(s33: str, *, start: str = "2018-01-01", fast: bool = False) -> dict:
    """1業種の全ステップを実行し結果 dict を返す。"""
    prof = SECTOR_PROFILES[s33]
    print(f"\n{'='*60}\n{s33} {prof.name} ({prof.kind})\n{'='*60}")

    panel = build_sector_daily_panel(s33, start=start)
    print(f"  パネル: {panel.index.min():%Y-%m-%d}〜{panel.index.max():%Y-%m-%d}  "
          f"{len(panel)}日 / 列{panel.shape[1]}")

    # Step 1: 因果グラフ
    graph = learn_pcmci_graph(panel, tau_max=5, pc_alpha=0.05)
    colliders = identify_colliders(graph, prof)
    # watch リストの collider も調整集合から除外
    adj_vars = [v for v in graph.adjustment_set.get("VALUE_to_RET", [])
                if v not in colliders]
    graph.adjustment_set["VALUE_to_RET"] = adj_vars
    print(f"  Collider 同定: {colliders or '（なし）'}")
    print(f"  調整集合 VALUE→RET: {adj_vars}")

    summ = graph_summary(graph)
    value_edges = summ[summ["is_value_edge"]] if not summ.empty else summ
    if not value_edges.empty:
        top = value_edges.head(3)
        for _, r in top.iterrows():
            print(f"    VALUE→RET lag={r['lag']} MCI={r['mci']:+.3f}")

    if not fast:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        plot_causal_graph(graph, str(REPORT_DIR / f"graph_{s33}.png"),
                          title=f"{s33} {prof.name}")

    # Step 2: 構造ブレイク（調整集合は Step1 の因果グラフから固定）
    adj_vars = graph.adjustment_set.get("VALUE_to_RET", [])
    step = 42 if fast else 21
    brk = detect_structural_breaks(panel, s33, window=252, step=step,
                                   adjustment_vars=adj_vars,
                                   skip_graph_diff=fast)
    cmp = compare_detection_methods(brk)
    print(f"  因果CUSUM警報: {len(brk.causal_alarms)}  "
          f"パフォCUSUM: {len(brk.perf_alarms)}  "
          f"グラフ差分: {len(brk.edge_diff_events)}")
    if not cmp.empty:
        print(cmp.to_string(index=False))

    # Step 3: 因果効果 + refutation
    effect = estimate_causal_effect(panel, graph, use_econml=not fast)
    refut = {} if fast else run_refutation_tests(panel, graph, n_simulations=10)
    print(f"  ATE(VALUE→RET): {effect.ate:+.6f}  CATE_mean: {effect.cate_mean}")
    print(f"  Refutation: {refut}")

    return {
        "s33": s33, "name": prof.name, "kind": prof.kind,
        "colliders": colliders,
        "adjustment": graph.adjustment_set.get("VALUE_to_RET", []),
        "causal_alarms": [str(d) for d in brk.causal_alarms],
        "perf_alarms": [str(d) for d in brk.perf_alarms],
        "graph_events": [str(d) for d in brk.edge_diff_events],
        "ate": effect.ate, "cate_mean": effect.cate_mean,
        "refutation": refut, "comparison": cmp.to_dict("records"),
    }


def write_final_report(results: list[dict]) -> None:
    """最終レポートを Markdown で出力。"""
    lines = [
        "# 32 — セクター因果的レジーム検知：最終レポート",
        "",
        "> 自動生成: `examples/causal_sector_pipeline.py`",
        "",
        "## 1. 概要（López de Prado 原則の適用）",
        "",
        "- **因果グラフ**: tigramite PCMCI+（ParCorr）でセクター別に学習。",
        "- **Collider 回避**: グラフ上で T→C←Y の変数を調整集合から除外。",
        "- **Structural break**: VALUE→RET エッジ強度のローリング推定 + CUSUM。",
        "  パフォーマンスブレイク（平均リターン変化）との先行性を比較。",
        "- **因果効果**: DoWhy backdoor.linear_regression + EconML LinearDML。",
        "- **Refutation**: placebo treatment, random common cause, data subset。",
        "",
        "## 2. セクター別結果",
        "",
        "| S33 | 業種 | 種別 | Collider | ATE | CATE | 因果警報数 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['s33']} | {r['name']} | {r['kind']} | "
            f"{', '.join(r['colliders']) or '—'} | "
            f"{r['ate']:+.4f} | "
            f"{r['cate_mean'] if r['cate_mean'] is not None else '—'} | "
            f"{len(r['causal_alarms'])} |"
        )

    lines += [
        "",
        "## 3. VALUE→RET 構造変化のサマリー",
        "",
    ]
    for r in results:
        lines += [
            f"### {r['s33']} {r['name']}",
            "",
            f"- 調整変数: `{r['adjustment']}`",
            f"- 因果CUSUM: {', '.join(r['causal_alarms'][:5]) or '—'}",
            f"- グラフ構造変化: {', '.join(r['graph_events'][:5]) or '—'}",
            f"- Refutation: `{r['refutation']}`",
            "",
        ]

    lines += [
        "## 4. 日本株イベントとの照合",
        "",
        "| イベント | 関連セクターでの因果警報 |",
        "|---|---|",
    ]
    for dt, ev in JP_STRUCTURAL_EVENTS:
        hits = []
        for r in results:
            for a in r["causal_alarms"] + r["graph_events"]:
                if abs((pd.Timestamp(a) - pd.Timestamp(dt)).days) < 90:
                    hits.append(r["s33"])
                    break
        lines.append(f"| {dt} {ev} | {', '.join(sorted(set(hits))) or '—'} |")

    lines += [
        "",
        "## 5. メタラベリング統合",
        "",
        "`examples/causal_sector_meta_gate.py` で value+PEAD 一次戦略に",
        "因果特徴量（edge強度・安定性・構造ブレイク近接度）を追加した",
        "二次モデルを walk-forward 学習。CPCV + DSR で比較。",
        "",
        "## 6. 限界と注意点",
        "",
        "- VALUE は月次→日次 ffill のため、日次 PCMCI の VALUE エッジは解釈に注意。",
        "- PCMCI+ は線形 ParCorr 前提。非線形・レジーム依存の因果は underfit しうる。",
        "- 全33業種の同時実行は計算コスト大（ローリング再学習がボトルネック）。",
        "- DoWhy refutation の p-value は標本依存。過度な最適化は避けること。",
        "",
        "## 7. 次のステップ（ライブ運用向け）",
        "",
        "1. 月次リバランス前に代表セクターの VALUE→RET エッジ強度を更新",
        "2. 因果CUSUM 警報時にメタゲートサイズを自動縮小",
        "3. Collider 出現（新エッジ）をアラートとして監視",
        "4. BOJ政策・PBR改革イベント後のグラフ再学習を四半期ごとに実施",
        "",
        f"可視化: `{REPORT_DIR}/graph_*.png`",
        "",
    ]
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n最終レポート: {FINAL_REPORT}")


def _write_all_sectors_report(results: list[dict], path: Path) -> None:
    """全33業種バッチ結果のサマリーレポート。"""
    df = pd.DataFrame(results)
    pos_ate = (df["ate"] > 0).sum()
    neg_ate = (df["ate"] < 0).sum()
    by_kind = df.groupby("kind").agg(
        n=("s33", "count"),
        mean_ate=("ate", "mean"),
        mean_alarms=("causal_alarms", lambda s: s.apply(len).mean()),
    ).reset_index()

    lines = [
        "# 36 — 全33業種 因果レジーム検知バッチ結果",
        "",
        f"> 自動生成: `causal_sector_pipeline.py --all-sectors --fast`",
        f"> 成功: {len(results)} 業種",
        "",
        "## 1. ATE(VALUE→RET) 分布",
        "",
        f"- 正（バリュー効果あり）: {pos_ate} 業種",
        f"- 負（逆/弱い）: {neg_ate} 業種",
        "",
        "## 2. 業種種別サマリー",
        "",
        "| kind | n | mean ATE | mean 因果警報数 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in by_kind.iterrows():
        lines.append(
            f"| {row['kind']} | {int(row['n'])} | {row['mean_ate']:+.6f} | "
            f"{row['mean_alarms']:.1f} |"
        )

    lines += [
        "",
        "## 3. 全業種一覧",
        "",
        "| S33 | 業種 | kind | ATE | 因果警報 | Collider |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in sorted(results, key=lambda x: x["s33"]):
        lines.append(
            f"| {r['s33']} | {r['name']} | {r['kind']} | {r['ate']:+.4f} | "
            f"{len(r['causal_alarms'])} | {', '.join(r['colliders'][:3]) or '—'} |"
        )

    lines += [
        "",
        "## 4. 構造変化が多い業種（因果警報上位5）",
        "",
    ]
    top = sorted(results, key=lambda x: len(x["causal_alarms"]), reverse=True)[:5]
    for r in top:
        lines.append(f"- **{r['s33']} {r['name']}**: {len(r['causal_alarms'])}件  ATE={r['ate']:+.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-sectors", action="store_true",
                        help="全33業種を実行（時間がかかる）")
    parser.add_argument("--fast", action="store_true",
                        help="全業種向け高速モード（refutation/可視化スキップ）")
    parser.add_argument("--start", default="2018-01-01")
    args = parser.parse_args()

    sectors = sorted(SECTOR_PROFILES.keys()) if args.all_sectors else list(REPRESENTATIVE_SECTORS)
    print("=" * 72)
    print(f"セクター因果的レジーム検知パイプライン（{len(sectors)} 業種）")
    print("=" * 72)

    results = []
    for s33 in sectors:
        try:
            results.append(run_sector(s33, start=args.start, fast=args.fast))
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {s33}: {e}")

    if results:
        if not args.all_sectors:
            write_final_report(results)
        else:
            all_report = ROOT / "docs" / "36-causal-sector-all33-results.md"
            _write_all_sectors_report(results, all_report)
            print(f"\n全業種レポート: {all_report}")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fname = "results_all33.json" if args.all_sectors else "results.json"
        pd.DataFrame(results).to_json(
            REPORT_DIR / fname, orient="records", force_ascii=False, indent=2)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())