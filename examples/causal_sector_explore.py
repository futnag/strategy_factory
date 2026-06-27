"""探索・データ理解レポート生成（33業種セクター因果分析の前提確認）。

data/README.md → docs/24-data-catalog.md に従いローカル data/ を走査し、
セクター別データ品質を評価する。出力: docs/31-causal-sector-data-exploration.md

実行: .venv\\Scripts\\python.exe examples\\causal_sector_explore.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invest_system.research.causal_sector.config import (  # noqa: E402
    REPRESENTATIVE_SECTORS, SECTOR_PROFILES, JP_STRUCTURAL_EVENTS,
)
from invest_system.research.causal_sector.panels import (  # noqa: E402
    sector_data_quality_report, build_sector_daily_panel,
)

OUT = ROOT / "docs" / "31-causal-sector-data-exploration.md"


def _df_to_md(df: pd.DataFrame) -> str:
    """tabulate 不要の簡易 Markdown 表。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c])[:40] for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    print("=" * 72)
    print("セクター因果分析 — データ探索レポート")
    print("=" * 72)

    # 全業種品質
    print("\n全33業種のデータ品質を評価中…")
    quality = sector_data_quality_report()
    if "error" in quality.columns:
        ok = quality[quality["error"].isna()]
    else:
        ok = quality
    print(f"  成功: {len(ok)} / {len(quality)} 業種")

    # 代表セクターのパネルサンプル
    samples = {}
    for s33 in REPRESENTATIVE_SECTORS:
        try:
            p = build_sector_daily_panel(s33, start="2020-01-01")
            samples[s33] = p
            prof = SECTOR_PROFILES[s33]
            print(f"  {s33} {prof.name}: {len(p)}日, 列={list(p.columns)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {s33}: ERROR {e}")

    # マクロ概要
    macro = pd.read_parquet(ROOT / "data" / "supplemental" / "macro_extended.parquet")
    macro = macro[~macro.index.duplicated(keep="last")]

    lines = [
        "# 31 — セクター因果分析：データ探索レポート",
        "",
        "> 自動生成: `examples/causal_sector_explore.py`",
        "",
        "## 1. リポジトリ構造（データ層）",
        "",
        "| 層 | パス | 用途 |",
        "|---|---|---|",
        "| L0 Raw | `data/jquants/`, `data/edinet/` | 日次株価・財務・空売り |",
        "| L1 Silver | `data/processed/equities/wide/` | Date×Code wide |",
        "| L2 Gold | `data/features/` | 派生特徴量（returns, vol, mom 等） |",
        "| マクロ | `data/supplemental/macro_extended.parquet` | 外生ドライバ（日次） |",
        "| 研究 | `data/phase4b/X.parquet` | book_to_market（月次×銘柄） |",
        "",
        "## 2. 利用可能な主要変数",
        "",
        "### 内生（セクター集約）",
        "- `RET`: 業種内等加重中央値日次リターン（`features/returns.parquet`）",
        "- `VALUE`: 業種 B/M 中央値（`phase4b/X` 月次 → 日次 ffill）",
        "- `VOL`, `MOM`: 業種集約ボラ・モメンタム",
        "- `SHORT_RATIO`: `jquants/short_ratio/s33_{code}.parquet`",
        "",
        "### 外生ドライバ（`macro_extended.parquet`）",
        f"- 期間: {macro.index.min():%Y-%m-%d} 〜 {macro.index.max():%Y-%m-%d}",
        f"- 列数: {macro.shape[1]}（金利・為替・VIX・コモディティ・米株指数）",
        "",
        "### セクター別ドライバ選定（`causal_sector/config.py`）",
        "",
        "| 種別 | 代表業種 | ドライバ |",
        "|---|---|---|",
    ]
    for kind in ("heavy_industry", "it_comm", "financial"):
        profs = [p for p in SECTOR_PROFILES.values() if p.kind == kind][:2]
        for p in profs:
            lines.append(f"| {kind} | {p.s33} {p.name} | {', '.join(p.drivers[:4])}… |")

    lines += [
        "",
        "## 3. 欠損状況・セクター別データ品質",
        "",
        _df_to_md(quality.round(4)),
        "",
        "## 4. 代表セクター・パネル統計（2020以降）",
        "",
    ]
    for s33, p in samples.items():
        prof = SECTOR_PROFILES[s33]
        desc = p.describe().T[["mean", "std", "min", "max"]].round(4).reset_index()
        desc = desc.rename(columns={"index": "var"})
        lines += [
            f"### {s33} {prof.name}（{prof.kind}）",
            "",
            _df_to_md(desc),
            "",
        ]

    lines += [
        "## 5. 日本株イベントアンカー（因果構造変化の参照）",
        "",
        "| 日付 | イベント |",
        "|---|---|",
    ]
    for dt, ev in JP_STRUCTURAL_EVENTS:
        lines.append(f"| {dt} | {ev} |")

    lines += [
        "",
        "## 6. 制約と注意",
        "",
        "- `VALUE` は月次開示ベースのため日次因果グラフでは**ゆっくり動く**変数。",
        "  VALUE→RET エッジはラグ付きリンクとして解釈すること。",
        "- セクター集約は等加重中央値（大型偏重を避け、業種特性を保持）。",
        "- Collider 候補: VOL, SHORT_RATIO, MOM（リターンとファクターの合流点になりうる）。",
        "- 全33業種の PCMCI+ は計算コスト大のため、代表6業種で検証後に拡張推奨。",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nレポート保存: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())