"""レジーム適応パターン全比較（株式・探索最良設定 + ブラックリスト/セクター除外）。

比較モード:
  none         - レジーム無効（参照）
  gate         - 高ボラブロック（従来 filtered）
  sizing       - 執行パラメータ動的調整
  exit         - イグジット動的調整
  entry        - エントリーシグナル動的調整
  sizing_exit  - 執行 + イグジット
  sizing_entry - 執行 + エントリー
  exit_entry   - イグジット + エントリー
  full         - 全層

実行: python examples/kaiten_regime_adapt_compare.py
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.research.kaiten import (  # noqa: E402
    CONFIG,
    REGIME_ADAPT_MODES,
    filtered_equity_config,
    load_symbols,
    run_backtest,
)

_COMPARE_MODES = [m for m in REGIME_ADAPT_MODES if m != "none"]
_PERIODS = [("full", None, None), ("oos_2020", "2020-01-01", None)]


def _cfg_for_mode(mode: str) -> dict:
    if mode == "none":
        cfg = filtered_equity_config(regime_adapt_mode="none")
        cfg["use_regime_filter"] = False
        cfg["regime_adapt_mode"] = "none"
        return cfg
    return filtered_equity_config(regime_adapt_mode=mode)


def _metrics(eq: pd.DataFrame, tr: pd.DataFrame, init: float) -> dict:
    final = eq["Equity"].iloc[-1]
    dd = (eq["Equity"] / eq["Equity"].cummax() - 1).min()
    n = len(tr) if not tr.empty else 0
    pf = float("nan")
    exp = float("nan")
    if n:
        w = tr[tr["pnl"] > 0]["pnl"].sum()
        l = -tr[tr["pnl"] <= 0]["pnl"].sum()
        pf = w / l if l > 0 else float("inf")
        exp = tr["pnl"].sum() / n
    return {
        "n_trades": n,
        "total_ret": final / init - 1,
        "max_dd": dd,
        "profit_factor": pf,
        "exp_per_trade": exp,
    }


def main() -> None:
    base_strategy = {
        "entry_method": "bb",
        "bb_sigma": 2.5,
        "trend_ma_period": 60,
        "use_ma_rise_filter": True,
        "ma_rise_periods": [5, 20, 60],
        "ma_rise_days": 7,
        "use_ma_order_filter": True,
    }

    rows: list[dict] = []
    modes = ["none"] + _COMPARE_MODES

    for period_label, start, end in _PERIODS:
        print(f"\n{'=' * 72}\n  period={period_label}\n{'=' * 72}")
        for mode in modes:
            cfg = _cfg_for_mode(mode)
            cfg.update(base_strategy)
            if start:
                cfg["start_date"] = start
            if end:
                cfg["end_date"] = end

            tag = f"{mode}_{period_label}"
            print(f"[RUN] {tag} ...", flush=True)
            data, mask = load_symbols(cfg)
            eq, tr = run_backtest(data, cfg, universe_mask=mask)
            m = _metrics(eq, tr, cfg["initial_capital"])
            rows.append({"mode": mode, "period": period_label, **m})
            print(
                f"  ret={m['total_ret']:+.2%} dd={m['max_dd']:.2%} "
                f"pf={m['profit_factor']:.2f} n={m['n_trades']}"
            )

    df = pd.DataFrame(rows)
    out = Path(CONFIG["output_dir"]) / "regime_adapt_compare.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n[INFO] 保存: {out}")
    for period_label in [p[0] for p in _PERIODS]:
        sub = df[df["period"] == period_label].copy()
        sub = sub.sort_values("total_ret", ascending=False)
        print(f"\n--- ランキング ({period_label}, total_ret) ---")
        print(sub[["mode", "n_trades", "total_ret", "max_dd", "profit_factor"]].to_string(index=False))

    # OOS 複合スコア: ret - 0.5*|dd| + 0.1*log1p(pf)
    oos = df[df["period"] == "oos_2020"].copy()
    oos["score"] = (
        oos["total_ret"]
        - 0.5 * oos["max_dd"].abs()
        + 0.1 * oos["profit_factor"].clip(upper=5).map(lambda x: __import__("math").log1p(x))
    )
    best = oos.sort_values("score", ascending=False).iloc[0]
    print(
        f"\n[BEST OOS] mode={best['mode']} "
        f"ret={best['total_ret']:+.2%} dd={best['max_dd']:.2%} pf={best['profit_factor']:.2f} "
        f"score={best['score']:.4f}"
    )


if __name__ == "__main__":
    main()