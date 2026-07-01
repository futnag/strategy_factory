"""MA 連続上昇 × パーフェクトオーダー パラメータ探索。

Phase 1: 2020-2024 でグリッド（データ1回読込）
Phase 2: 上位候補を全期間で再検証

実行: python examples/kaiten_ma_rise_sweep.py
"""
from __future__ import annotations

import ast
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.research.kaiten import CONFIG  # noqa: E402
from invest_system.research.kaiten.data import load_symbols  # noqa: E402
from invest_system.research.kaiten.sweep import run_single  # noqa: E402


def _base() -> dict:
    cfg = deepcopy(CONFIG)
    cfg.update({
        "entry_method": "bb",
        "bb_sigma": 2.5,
        "trend_ma_period": 60,
        "use_ma_rise_filter": True,
    })
    cfg["start_date"] = get_env("KAITEN_START", "2020-01-01") or "2020-01-01"
    cfg["end_date"] = get_env("KAITEN_END", "2024-12-31") or "2024-12-31"
    return cfg


def _print_top(df: pd.DataFrame, title: str, n: int = 10) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")
    cols = [
        "ma_rise_periods", "ma_rise_days", "use_ma_order_filter",
        "n_trades", "total_ret", "max_dd", "sharpe", "profit_factor",
        "pct_timeout", "pct_target", "pct_stop",
    ]
    show = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 200)
    print(df[show].head(n).to_string(index=False))


def main() -> None:
    out = Path(CONFIG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    base = _base()

    period_sets = [(5, 20), (10, 30), (20, 60), (5, 20, 60), (5, 10, 20)]
    rise_days_opts = [3, 5, 7]
    order_opts = [False, True]

    print(f"[INFO] Phase1 期間: {base['start_date']} 〜 {base['end_date']}", flush=True)
    print("[INFO] データ読込...", flush=True)
    data, mask = load_symbols(base)

    rows: list[dict] = []
    combos = list(product(period_sets, rise_days_opts, order_opts))
    for i, (periods, days, use_order) in enumerate(combos):
        cfg = deepcopy(base)
        cfg["ma_rise_periods"] = list(periods)
        cfg["ma_rise_days"] = days
        cfg["use_ma_order_filter"] = use_order
        cfg["ma_order_periods"] = list(periods)
        tag = f"p{periods}_d{days}_ord{int(use_order)}"
        print(f"[{i + 1}/{len(combos)}] {tag}", flush=True)
        _, _, sm = run_single(cfg, data, mask)
        rows.append(sm)

    res = pd.DataFrame(rows).sort_values(
        ["profit_factor", "sharpe", "total_ret"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    p1_path = out / "param_sweep_ma_rise.csv"
    res.to_csv(p1_path, index=False, encoding="utf-8-sig")
    _print_top(res, "Phase 1: MA rise × order（探索期間・PF順）")

    # baseline: rise filter off
    cfg_off = deepcopy(base)
    cfg_off["use_ma_rise_filter"] = False
    cfg_off["use_ma_order_filter"] = False
    _, _, sm_off = run_single(cfg_off, data, mask)
    print("\n--- baseline (MA rise OFF) ---")
    print(f"  trades={sm_off['n_trades']} ret={sm_off['total_ret']:+.2%} PF={sm_off['profit_factor']:.3f}")

    # Phase 2: top 5 full period
    print("\n[INFO] Phase 2: 上位5件を全期間で再検証...", flush=True)
    full_rows: list[dict] = []
    for _, row in res.head(5).iterrows():
        cfg = deepcopy(CONFIG)
        cfg.update({
            "entry_method": "bb", "bb_sigma": 2.5, "trend_ma_period": 60,
            "use_ma_rise_filter": True,
            "ma_rise_periods": ast.literal_eval(row["ma_rise_periods"]),
            "ma_rise_days": int(row["ma_rise_days"]),
            "use_ma_order_filter": bool(row["use_ma_order_filter"]),
            "ma_order_periods": ast.literal_eval(row["ma_rise_periods"]),
            "start_date": None, "end_date": None,
        })
        data_f, mask_f = load_symbols(cfg)
        _, _, sm = run_single(cfg, data_f, mask_f)
        sm["label"] = (
            f"{row['ma_rise_periods']}_d{int(row['ma_rise_days'])}"
            f"_ord{int(row['use_ma_order_filter'])}"
        )
        full_rows.append(sm)

    cfg_off_f = deepcopy(CONFIG)
    cfg_off_f.update({
        "entry_method": "bb", "bb_sigma": 2.5, "trend_ma_period": 60,
        "use_ma_rise_filter": False, "use_ma_order_filter": False,
        "start_date": None, "end_date": None,
    })
    data_f, mask_f = load_symbols(cfg_off_f)
    _, _, sm_b = run_single(cfg_off_f, data_f, mask_f)
    sm_b["label"] = "baseline_no_rise"
    full_rows.append(sm_b)

    full = pd.DataFrame(full_rows).sort_values("profit_factor", ascending=False)
    p2_path = out / "param_sweep_ma_rise_full.csv"
    full.to_csv(p2_path, index=False, encoding="utf-8-sig")
    _print_top(full, "Phase 2: 全期間 OOS 再検証")

    print(f"\n[INFO] 保存: {p1_path}, {p2_path}")


if __name__ == "__main__":
    main()