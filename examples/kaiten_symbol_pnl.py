"""銘柄別 PnL 集計（共有ポートフォリオのトレード履歴から）。"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.research.kaiten import CONFIG, run_backtest
from invest_system.research.kaiten.data import load_symbols


def symbol_summary(tr: pd.DataFrame) -> pd.DataFrame:
    return tr.groupby("symbol").agg(
        n_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        avg_ret=("ret_pct", "mean"),
    )


def report(label: str, cfg: dict) -> pd.DataFrame:
    data, mask = load_symbols(cfg)
    _, tr = run_backtest(data, cfg, universe_mask=mask)
    if tr.empty:
        print(f"{label}: トレードなし")
        return pd.DataFrame()

    by = symbol_summary(tr).sort_values("total_pnl", ascending=False)
    pos = by[by["total_pnl"] > 0]
    neg = by[by["total_pnl"] <= 0]

    print(f"\n=== {label} ===")
    print(f"トレード数: {len(tr)} / 銘柄数: {len(by)}")
    print(f"プラス銘柄: {len(pos)} ({len(pos) / len(by):.1%})")
    print(f"マイナス銘柄: {len(neg)} ({len(neg) / len(by):.1%})")
    print(f"プラス銘柄 PnL 合計: {pos['total_pnl'].sum():,.0f}")
    print(f"マイナス銘柄 PnL 合計: {neg['total_pnl'].sum():,.0f}")
    print(f"ネット PnL: {by['total_pnl'].sum():,.0f}")
    print("\n上位10:")
    print(by.head(10).to_string())
    return by


def main() -> None:
    out = Path(CONFIG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    cfg_def = deepcopy(CONFIG)
    by_def = report("既定 atr2.5_ma200", cfg_def)
    if not by_def.empty:
        by_def.to_csv(out / "symbol_pnl_default.csv", encoding="utf-8-sig")

    cfg_bb = deepcopy(CONFIG)
    cfg_bb.update({"entry_method": "bb", "bb_sigma": 2.5, "trend_ma_period": 60})
    by_bb = report("探索最良 bb2.5_ma60", cfg_bb)
    if not by_bb.empty:
        by_bb.to_csv(out / "symbol_pnl_bb_ma60.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()