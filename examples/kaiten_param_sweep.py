"""回転手法パラメータ探索（グリッドサーチ）。

データは1回だけ読み込み、複数 CONFIG を順次バックテストする。
結果は output/kaiten/param_sweep_*.csv に保存。

実行:
  python examples/kaiten_param_sweep.py

環境変数:
  KAITEN_START / KAITEN_END  探索期間（既定 2020-01-01 〜 2024-12-31）
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

from invest_system.config import get_env  # noqa: E402
from invest_system.research.kaiten import CONFIG  # noqa: E402
from invest_system.research.kaiten.data import load_symbols  # noqa: E402
from invest_system.research.kaiten.sweep import run_single, sweep_grid  # noqa: E402


def _base() -> dict:
    cfg = deepcopy(CONFIG)
    cfg["start_date"] = get_env("KAITEN_START", "2020-01-01") or "2020-01-01"
    cfg["end_date"] = get_env("KAITEN_END", "2024-12-31") or "2024-12-31"
    return cfg


def _print_top(df: pd.DataFrame, title: str, n: int = 8) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)
    cols = [
        "entry_method", "atr_mult", "bb_sigma", "trend_ma_period",
        "take_profit_pct", "max_hold_days", "n_trades",
        "total_ret", "cagr", "max_dd", "sharpe", "profit_factor",
        "pct_timeout", "pct_target", "pct_stop",
    ]
    show = [c for c in cols if c in df.columns]
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.0f}")
    print(df[show].head(n).to_string(index=False))


def main() -> None:
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _base()

    print(f"[INFO] 探索期間: {base['start_date']} 〜 {base['end_date']}")
    print("[INFO] データ読込中（1回のみ）...", flush=True)
    data, mask = load_symbols(base)
    print(f"[INFO] 銘柄数: {len(data)}")

    # --- Phase 1: ATR × トレンド MA ---
    atr_grid = {
        "entry_method": ["atr"],
        "atr_mult": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
        "trend_ma_period": [40, 60, 100, 150, 200],
    }
    atr_res = sweep_grid(atr_grid, base, data=data, universe_mask=mask)
    atr_path = out_dir / "param_sweep_atr.csv"
    atr_res.to_csv(atr_path, index=False, encoding="utf-8-sig")
    _print_top(atr_res, "Phase 1: ATR × trend_ma_period（PF順）")

    # --- Phase 2: BB × トレンド MA ---
    bb_grid = {
        "entry_method": ["bb"],
        "bb_sigma": [1.5, 2.0, 2.5, 3.0],
        "trend_ma_period": [40, 60, 100, 150, 200],
    }
    bb_res = sweep_grid(bb_grid, base, data=data, universe_mask=mask)
    bb_path = out_dir / "param_sweep_bb.csv"
    bb_res.to_csv(bb_path, index=False, encoding="utf-8-sig")
    _print_top(bb_res, "Phase 2: BB sigma × trend_ma_period（PF順）")

    # --- Phase 3: イグジット（Phase1 上位の ATR 設定で） ---
    best_atr = atr_res.iloc[0]
    exit_base = deepcopy(base)
    exit_base.update({
        "entry_method": "atr",
        "atr_mult": float(best_atr["atr_mult"]),
        "trend_ma_period": int(best_atr["trend_ma_period"]),
    })
    exit_grid = {
        "take_profit_pct": [0.05, 0.07, 0.10],
        "max_hold_days": [3, 5, 7],
    }
    exit_res = sweep_grid(exit_grid, exit_base, data=data, universe_mask=mask)
    exit_path = out_dir / "param_sweep_exit.csv"
    exit_res.to_csv(exit_path, index=False, encoding="utf-8-sig")
    _print_top(exit_res, f"Phase 3: イグジット（base=atr{best_atr['atr_mult']}_ma{int(best_atr['trend_ma_period'])})")

    # --- Phase 4: ベスト候補を全期間で再検証 ---
    best_exit = exit_res.iloc[0]
    full_cfg = deepcopy(base)
    full_cfg.update({
        "entry_method": "atr",
        "atr_mult": float(best_atr["atr_mult"]),
        "trend_ma_period": int(best_atr["trend_ma_period"]),
        "take_profit_pct": float(best_exit["take_profit_pct"]),
        "max_hold_days": int(best_exit["max_hold_days"]),
        "start_date": None,
        "end_date": None,
    })
    print("\n[INFO] Phase 4: ベスト候補を全期間（2016〜）で再検証...", flush=True)
    data_full, mask_full = load_symbols(full_cfg)
    _, tr_full, sm_full = run_single(full_cfg, data_full, mask_full)
    sm = pd.DataFrame([sm_full])
    _print_top(sm, "Phase 4: 全期間 OOS 再検証（探索期間外）")
    sm.to_csv(out_dir / "param_sweep_best_full.csv", index=False, encoding="utf-8-sig")

    print(f"\n[INFO] 保存: {atr_path}, {bb_path}, {exit_path}")


if __name__ == "__main__":
    main()