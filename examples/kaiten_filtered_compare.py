"""レジーム + ブラックリスト ON/OFF 比較（株式・探索最良設定）。

ブラックリスト・セクター除外・パラメータは 2016-2022 の探索由来 → 真の OOS は 2023 以降。
"full" は in-sample を含む参考値。採否の判断材料には oos_2023 のみを使うこと。
（旧 "oos_2020" は構築期間 2016-2022 と3年重複しており OOS でなかった）

実行:
  python examples/kaiten_build_blacklist.py   # 先に実行推奨
  python examples/kaiten_filtered_compare.py
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
    filtered_equity_config,
    load_symbols,
    performance_report,
    run_backtest,
)


def main_cli() -> None:
    base = deepcopy(CONFIG)
    base.update({
        "entry_method": "bb",
        "bb_sigma": 2.5,
        "trend_ma_period": 60,
        "use_ma_rise_filter": True,
        "ma_rise_periods": [5, 20, 60],
        "ma_rise_days": 7,
        "use_ma_order_filter": True,
        "use_regime_filter": False,
        "use_symbol_blacklist": False,
        "exclude_sectors": [],
    })
    filt = filtered_equity_config()

    periods = [
        ("full", None, None),               # 参考値（探索期間 2016-2022 を含む＝OOS でない）
        ("oos_2023", "2023-01-01", None),   # 真の OOS（ブラックリスト構築期間と非重複）
    ]

    rows: list[dict] = []
    for period_label, start, end in periods:
        for label, cfg in [("baseline", base), ("filtered", filt)]:
            run_cfg = deepcopy(cfg)
            if start:
                run_cfg["start_date"] = start
            if end:
                run_cfg["end_date"] = end
            tag = f"{label}_{period_label}"
            print(f"\n{'#' * 72}\n  {tag}\n{'#' * 72}")
            if label == "filtered":
                bl = Path(run_cfg.get("symbol_blacklist_path") or "")
                print(f"[INFO] レジーム: vol_regime<={run_cfg['regime_vol_max']}")
                print(f"[INFO] 除外セクター: {run_cfg['exclude_sectors']}")
                print(f"[INFO] ブラックリスト: {bl} ({'あり' if bl.exists() else 'なし'})")
            data, mask = load_symbols(run_cfg)
            eq, tr = run_backtest(data, run_cfg, universe_mask=mask)
            performance_report(eq, tr, run_cfg)
            init = run_cfg["initial_capital"]
            final = eq["Equity"].iloc[-1]
            dd = (eq["Equity"] / eq["Equity"].cummax() - 1).min()
            n = len(tr) if not tr.empty else 0
            pf = float("nan")
            if n:
                w = tr[tr["pnl"] > 0]["pnl"].sum()
                l = -tr[tr["pnl"] <= 0]["pnl"].sum()
                pf = w / l if l > 0 else float("inf")
            rows.append({
                "label": tag,
                "period": period_label,
                "variant": label,
                "n_trades": n,
                "total_ret": final / init - 1,
                "max_dd": dd,
                "profit_factor": pf,
            })

    out = Path(CONFIG["output_dir"]) / "filtered_compare.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 比較保存: {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main_cli()