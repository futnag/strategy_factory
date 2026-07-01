"""日経225先物・TOPIX先物 限定のパラメータ探索（BB + ATR）。

対象: nk225_fut, topix_fut のみ。
各エントリー方式ごとに: エントリー → イグジット → 指値有効日数 → MA rise
Phase 2: BB/ATR 各トラック上位を全期間で再検証し比較。

実行:
  python examples/kaiten_futures_param_sweep.py
  $env:KAITEN_FUT="nk225_fut"
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
from invest_system.research.kaiten import TUNABLE_PARAMS, futures_config  # noqa: E402
from invest_system.research.kaiten.data import load_symbols  # noqa: E402
from invest_system.research.kaiten.sweep import run_single, sweep_grid  # noqa: E402

FUTURES_KEYS = ("nk225_fut", "topix_fut")


def _base(key: str) -> dict:
    cfg = futures_config(key)
    cfg["start_date"] = get_env("KAITEN_START", "2020-01-01") or "2020-01-01"
    cfg["end_date"] = get_env("KAITEN_END", "2024-12-31") or "2024-12-31"
    return cfg


def _print_top(df: pd.DataFrame, title: str, n: int = 8) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")
    cols = [
        "external_key", "entry_method", "atr_mult", "bb_sigma", "trend_ma_period",
        "limit_valid_days", "stop_loss_pct", "take_profit_pct", "max_hold_days",
        "ma_rise_periods", "ma_rise_days", "use_ma_order_filter", "label",
        "n_trades", "total_ret", "max_dd", "sharpe", "profit_factor",
        "pct_timeout", "pct_target", "pct_stop",
    ]
    show = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 240)
    print(df[show].head(n).to_string(index=False))


def _entry_label(cfg: dict) -> str:
    if cfg["entry_method"] == "atr":
        return f"atr{cfg['atr_mult']}_ma{cfg['trend_ma_period']}"
    return f"bb{cfg['bb_sigma']}_ma{cfg['trend_ma_period']}"


def _sweep_ma_rise(base: dict, data: dict, mask) -> pd.DataFrame:
    period_sets = [(5, 20), (20, 60), (5, 20, 60), (5, 10, 20)]
    rise_days_opts = [3, 5, 7]
    order_opts = [False, True]
    rows: list[dict] = []
    for periods, days, use_order in product(period_sets, rise_days_opts, order_opts):
        cfg = deepcopy(base)
        cfg.update({
            "use_ma_rise_filter": True,
            "ma_rise_periods": list(periods),
            "ma_rise_days": days,
            "use_ma_order_filter": use_order,
            "ma_order_periods": list(periods),
        })
        _, _, sm = run_single(cfg, data, mask)
        rows.append(sm)
    return pd.DataFrame(rows).sort_values(
        ["profit_factor", "sharpe", "total_ret"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _run_track(
    key: str,
    base: dict,
    data: dict,
    mask,
    out: Path,
    entry_res: pd.DataFrame,
    tag: str,
) -> dict:
    """1エントリー方式（bb/atr）の exit → valid → ma_rise パイプライン。"""
    best = entry_res.iloc[0]
    track_base = deepcopy(base)
    track_base["entry_method"] = tag
    track_base["trend_ma_period"] = int(best["trend_ma_period"])
    if tag == "atr":
        track_base["atr_mult"] = float(best["atr_mult"])
    else:
        track_base["bb_sigma"] = float(best["bb_sigma"])

    lbl = _entry_label(track_base)
    print(f"\n--- {key} / {tag} track (base={lbl}) ---", flush=True)

    exit_grid = {
        "stop_loss_pct": [0.05, 0.07, 0.10],
        "take_profit_pct": [0.05, 0.07, 0.10],
        "max_hold_days": [3, 5, 7],
    }
    exit_res = sweep_grid(exit_grid, track_base, data=data, universe_mask=mask, verbose=False)
    exit_path = out / f"param_sweep_exit_{tag}_{key}.csv"
    exit_res.to_csv(exit_path, index=False, encoding="utf-8-sig")
    _print_top(exit_res, f"{key}/{tag}: イグジット")

    best_exit = exit_res.iloc[0]
    valid_base = deepcopy(track_base)
    valid_base.update({
        "stop_loss_pct": float(best_exit["stop_loss_pct"]),
        "take_profit_pct": float(best_exit["take_profit_pct"]),
        "max_hold_days": int(best_exit["max_hold_days"]),
    })

    valid_grid = {"limit_valid_days": [3, 5, 7, 10, 15]}
    valid_res = sweep_grid(valid_grid, valid_base, data=data, universe_mask=mask, verbose=False)
    valid_path = out / f"param_sweep_limit_valid_{tag}_{key}.csv"
    valid_res.to_csv(valid_path, index=False, encoding="utf-8-sig")
    _print_top(valid_res, f"{key}/{tag}: limit_valid_days")

    best_valid = valid_res.iloc[0]
    rise_base = deepcopy(valid_base)
    rise_base["limit_valid_days"] = int(best_valid["limit_valid_days"])

    rise_res = _sweep_ma_rise(rise_base, data, mask)
    rise_path = out / f"param_sweep_ma_rise_{tag}_{key}.csv"
    rise_res.to_csv(rise_path, index=False, encoding="utf-8-sig")
    _print_top(rise_res, f"{key}/{tag}: MA rise × order")

    cfg_off = deepcopy(rise_base)
    cfg_off["use_ma_rise_filter"] = False
    cfg_off["use_ma_order_filter"] = False
    _, _, sm_off = run_single(cfg_off, data, mask)
    print(f"  baseline (MA rise OFF): trades={sm_off['n_trades']} "
          f"ret={sm_off['total_ret']:+.2%} PF={sm_off['profit_factor']:.3f}")

    return {
        "tag": tag,
        "track_base": track_base,
        "best_exit": best_exit,
        "best_valid": best_valid,
        "rise_res": rise_res,
        "baseline": sm_off,
    }


def _full_validate(key: str, track: dict, top_n: int = 3) -> list[dict]:
    """探索期間ベストを全期間で再検証。"""
    tb = track["track_base"]
    be = track["best_exit"]
    bv = track["best_valid"]
    rise_res = track["rise_res"]
    rows: list[dict] = []

    for _, row in rise_res.head(top_n).iterrows():
        cfg = futures_config(key)
        cfg.update({
            "entry_method": tb["entry_method"],
            "trend_ma_period": int(tb["trend_ma_period"]),
            "stop_loss_pct": float(be["stop_loss_pct"]),
            "take_profit_pct": float(be["take_profit_pct"]),
            "max_hold_days": int(be["max_hold_days"]),
            "limit_valid_days": int(bv["limit_valid_days"]),
            "use_ma_rise_filter": True,
            "ma_rise_periods": ast.literal_eval(row["ma_rise_periods"]),
            "ma_rise_days": int(row["ma_rise_days"]),
            "use_ma_order_filter": bool(row["use_ma_order_filter"]),
            "ma_order_periods": ast.literal_eval(row["ma_rise_periods"]),
            "start_date": None,
            "end_date": None,
        })
        if tb["entry_method"] == "atr":
            cfg["atr_mult"] = float(tb["atr_mult"])
        else:
            cfg["bb_sigma"] = float(tb["bb_sigma"])
        data_f, mask_f = load_symbols(cfg)
        _, _, sm = run_single(cfg, data_f, mask_f)
        sm["label"] = (
            f"{track['tag']}_{row['ma_rise_periods']}_d{int(row['ma_rise_days'])}"
            f"_ord{int(row['use_ma_order_filter'])}"
        )
        rows.append(sm)

    cfg_off = futures_config(key)
    cfg_off.update({
        "entry_method": tb["entry_method"],
        "trend_ma_period": int(tb["trend_ma_period"]),
        "stop_loss_pct": float(be["stop_loss_pct"]),
        "take_profit_pct": float(be["take_profit_pct"]),
        "max_hold_days": int(be["max_hold_days"]),
        "limit_valid_days": int(bv["limit_valid_days"]),
        "use_ma_rise_filter": False,
        "use_ma_order_filter": False,
        "start_date": None,
        "end_date": None,
    })
    if tb["entry_method"] == "atr":
        cfg_off["atr_mult"] = float(tb["atr_mult"])
    else:
        cfg_off["bb_sigma"] = float(tb["bb_sigma"])
    data_f, mask_f = load_symbols(cfg_off)
    _, _, sm_b = run_single(cfg_off, data_f, mask_f)
    sm_b["label"] = f"{track['tag']}_baseline_no_rise"
    rows.append(sm_b)
    return rows


def sweep_one(key: str, out: Path) -> pd.DataFrame:
    base = _base(key)
    print(f"\n{'#' * 72}\n  {key}  Phase1: {base['start_date']} 〜 {base['end_date']}\n{'#' * 72}")
    print("[INFO] データ読込...", flush=True)
    data, mask = load_symbols(base)
    if not data:
        print(f"[WARN] {key}: データなし")
        return pd.DataFrame()

    # --- ATR × trend MA ---
    atr_grid = {
        "entry_method": ["atr"],
        "atr_mult": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
        "trend_ma_period": [40, 60, 100, 150],
    }
    atr_res = sweep_grid(atr_grid, base, data=data, universe_mask=mask)
    atr_path = out / f"param_sweep_atr_{key}.csv"
    atr_res.to_csv(atr_path, index=False, encoding="utf-8-sig")
    _print_top(atr_res, f"{key}: ATR mult × trend_ma（PF順）")

    # --- BB × trend MA ---
    bb_grid = {
        "entry_method": ["bb"],
        "bb_sigma": [1.5, 2.0, 2.5, 3.0],
        "trend_ma_period": [40, 60, 100, 150],
    }
    bb_res = sweep_grid(bb_grid, base, data=data, universe_mask=mask)
    bb_path = out / f"param_sweep_bb_{key}.csv"
    bb_res.to_csv(bb_path, index=False, encoding="utf-8-sig")
    _print_top(bb_res, f"{key}: BB σ × trend_ma（PF順）")

    atr_track = _run_track(key, base, data, mask, out, atr_res, "atr")
    bb_track = _run_track(key, base, data, mask, out, bb_res, "bb")

    # --- Phase 2: 全期間再検証（BB/ATR 各トラック） ---
    print(f"\n[INFO] {key} Phase2: BB/ATR 上位を全期間で再検証...", flush=True)
    full_rows = _full_validate(key, atr_track) + _full_validate(key, bb_track)
    full = pd.DataFrame(full_rows).sort_values("profit_factor", ascending=False)
    full_path = out / f"param_sweep_full_{key}.csv"
    full.to_csv(full_path, index=False, encoding="utf-8-sig")
    _print_top(full, f"{key}: 全期間 OOS（BB vs ATR）")

    # エントリー方式比較（各トラックの Phase1 MA-rise 最良）
    compare_rows = []
    for track in (atr_track, bb_track):
        best_rise = track["rise_res"].iloc[0].to_dict()
        best_rise["track"] = track["tag"]
        best_rise["phase"] = "phase1_explore"
        compare_rows.append(best_rise)
    for _, row in full.head(2).iterrows():
        r = row.to_dict()
        r["phase"] = "phase2_full"
        compare_rows.append(r)

    compare = pd.DataFrame(compare_rows)
    compare_path = out / f"param_sweep_entry_compare_{key}.csv"
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 保存: {atr_path}, {bb_path}, {full_path}, {compare_path}")
    return compare


def main() -> None:
    out = Path(futures_config("nk225_fut")["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 操作可能パラメータ数: {len(TUNABLE_PARAMS)}")
    print("[INFO] 探索: ATR(6×4) + BB(4×4) → exit(27) → valid(5) → MA rise(24) / 銘柄")

    fut = get_env("KAITEN_FUT")
    keys = (fut,) if fut else FUTURES_KEYS

    summaries: list[pd.DataFrame] = []
    for key in keys:
        summaries.append(sweep_one(key, out))

    if summaries:
        all_sum = pd.concat([s for s in summaries if not s.empty], ignore_index=True)
        path = out / "param_sweep_summary.csv"
        all_sum.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] 統合サマリー: {path}")


if __name__ == "__main__":
    main()