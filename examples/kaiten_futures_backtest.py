"""日経225先物・TOPIX先物への回転手法シミュレーション（探索最良パラメータ）。

既定:
  nk225_fut → BB（PF/DD 優先）  ※ KAITEN_ENTRY=atr で ATR 版
  topix_fut → BB のみ

実行:
  python examples/kaiten_futures_backtest.py
  $env:KAITEN_FUT="nk225_fut"; $env:KAITEN_ENTRY="atr"; python examples/kaiten_futures_backtest.py
  $env:KAITEN_COMPARE="1"; python examples/kaiten_futures_backtest.py  # 日経 BB/ATR 比較
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.research.kaiten import (  # noqa: E402
    futures_config,
    futures_entry_methods,
    main,
)

_DEFAULT_KEYS = ("nk225_fut", "topix_fut")


def _cfg(key: str, entry: str | None = None) -> dict:
    cfg = futures_config(key, entry=entry)
    if start := get_env("KAITEN_START"):
        cfg["start_date"] = start
    if end := get_env("KAITEN_END"):
        cfg["end_date"] = end
    if entry_env := get_env("KAITEN_ENTRY"):
        if entry is None:
            cfg = futures_config(key, entry=entry_env)
            if start := get_env("KAITEN_START"):
                cfg["start_date"] = start
            if end := get_env("KAITEN_END"):
                cfg["end_date"] = end
    if mult := get_env("KAITEN_CONTRACT_MULT"):
        cfg["contract_multiplier"] = float(mult)
    if cap := get_env("KAITEN_CAPITAL"):
        cfg["initial_capital"] = float(cap)
    return cfg


def _run_one(key: str, entry: str | None = None) -> dict | None:
    cfg = _cfg(key, entry=entry)
    print(f"\n{'#' * 72}\n  {cfg['external_key']} / {cfg['entry_method']}\n{'#' * 72}")
    eq, tr = main(cfg)
    if eq is None or eq.empty:
        print(f"[WARN] {key}: データ不足またはシミュレーション失敗")
        return None
    init = cfg["initial_capital"]
    final = eq["Equity"].iloc[-1]
    dd = (eq["Equity"] / eq["Equity"].cummax() - 1).min()
    n = len(tr) if tr is not None and not tr.empty else 0
    pf = float("nan")
    if n and not tr.empty:
        wins = tr[tr["pnl"] > 0]["pnl"].sum()
        losses = -tr[tr["pnl"] <= 0]["pnl"].sum()
        pf = wins / losses if losses > 0 else float("inf")
    return {
        "key": key,
        "entry_method": cfg["entry_method"],
        "label": cfg.get("instrument_label", key),
        "n_trades": n,
        "total_ret": final / init - 1,
        "max_dd": dd,
        "profit_factor": pf,
        "final_equity": final,
    }


def main_cli() -> None:
    fut = get_env("KAITEN_FUT")
    keys = (fut,) if fut else _DEFAULT_KEYS
    compare = get_env("KAITEN_COMPARE", "0") in ("1", "true", "True")
    rows: list[dict] = []

    for key in keys:
        if compare and key == "nk225_fut":
            for entry in futures_entry_methods(key):
                r = _run_one(key, entry=entry)
                if r:
                    rows.append(r)
        else:
            r = _run_one(key)
            if r:
                rows.append(r)

    if rows:
        import pandas as pd

        out = Path(futures_config("nk225_fut")["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(rows)
        path = out / "futures_summary.csv"
        summary.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] サマリー保存: {path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main_cli()