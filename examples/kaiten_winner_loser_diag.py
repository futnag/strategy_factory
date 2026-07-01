"""勝ち銘柄 vs 負け銘柄の特性比較（bb2.5_ma60・全期間）。"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import jquants as jq
from invest_system.data.store import load_wide
from invest_system.research.kaiten import CONFIG, run_backtest
from invest_system.research.kaiten.data import load_symbols


def _load_feature(name: str) -> pd.DataFrame:
    fp = Path("data/features") / f"{name}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index)
    df.columns = [str(c) for c in df.columns]
    return df


def _symbol_median(panel: pd.DataFrame, codes: list[str]) -> pd.Series:
    sub = panel.reindex(columns=[c for c in codes if c in panel.columns])
    return sub.median(axis=0, skipna=True)


def main() -> None:
    cfg = deepcopy(CONFIG)
    cfg.update({"entry_method": "bb", "bb_sigma": 2.5, "trend_ma_period": 60})

    print("[INFO] バックテスト実行...", flush=True)
    data, mask = load_symbols(cfg)
    _, tr = run_backtest(data, cfg, universe_mask=mask)

    sym_pnl = tr.groupby("symbol").agg(
        n_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        avg_hold=("hold_days", "mean"),
    )
    sym_pnl["group"] = np.where(sym_pnl["total_pnl"] > 0, "winner", "loser")

    # イグジット内訳（銘柄別 timeout 率）
    reason = tr.assign(
        base_reason=tr["reason"].str.replace(r"\(.*\)", "", regex=True)
    )
    timeout_rate = (
        reason.groupby("symbol")["base_reason"]
        .apply(lambda s: (s == "timeout").mean())
        .rename("pct_timeout")
    )
    target_rate = (
        reason.groupby("symbol")["base_reason"]
        .apply(lambda s: (s == "target").mean())
        .rename("pct_target")
    )
    sym_pnl = sym_pnl.join(timeout_rate).join(target_rate)

    codes = sym_pnl.index.astype(str).tolist()
    listed = jq.fetch_listed_info()
    listed = listed.drop_duplicates("Code", keep="last").set_index("Code")
    sym_pnl["sector"] = sym_pnl.index.map(
        lambda c: listed.at[c, "S33Nm"] if c in listed.index else None
    )
    sym_pnl["scale"] = sym_pnl.index.map(
        lambda c: listed.at[c, "ScaleCat"] if c in listed.index else None
    )

    turnover = load_wide("turnover")
    close = load_wide("adj_close")
    sym_pnl["med_turnover"] = sym_pnl.index.map(
        lambda c: turnover[c].median() if c in turnover.columns else np.nan
    )
    sym_pnl["med_price"] = sym_pnl.index.map(
        lambda c: close[c].median() if c in close.columns else np.nan
    )

    for feat, col in [
        ("rvol_60", "med_rvol60"),
        ("garman_klass_vol", "med_gk_vol"),
        ("amihud_illiq", "med_amihud"),
        ("beta", "med_beta"),
        ("high_52w", "med_high52w"),
        ("mom_3m", "med_mom3m"),
    ]:
        panel = _load_feature(feat)
        if panel.empty:
            continue
        med = _symbol_median(panel, codes)
        sym_pnl[col] = sym_pnl.index.map(lambda c, m=med: m.get(c, np.nan))

    compare_cols = [
        "n_trades", "win_rate", "avg_hold", "pct_timeout", "pct_target",
        "med_turnover", "med_price", "med_rvol60", "med_gk_vol",
        "med_amihud", "med_beta", "med_high52w", "med_mom3m",
    ]
    compare_cols = [c for c in compare_cols if c in sym_pnl.columns]

    win = sym_pnl[sym_pnl["group"] == "winner"]
    lose = sym_pnl[sym_pnl["group"] == "loser"]

    print("\n=== 勝ち vs 負け：数値特性（中央値比較） ===")
    rows = []
    for c in compare_cols:
        w = win[c].median()
        l = lose[c].median()
        rows.append({"metric": c, "winner_med": w, "loser_med": l, "ratio_w_l": w / l if l else np.nan})
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n=== セクター別：勝ち銘柄比率（S33Nm・取引5回以上） ===")
    active = sym_pnl[sym_pnl["n_trades"] >= 5].copy()
    sec = active.groupby("sector").agg(
        n=("group", "count"),
        win_rate=("group", lambda s: (s == "winner").mean()),
        avg_pnl=("total_pnl", "mean"),
    ).sort_values("win_rate", ascending=False)
    print(sec.head(12).to_string())
    print("...")
    print(sec.tail(8).to_string())

    print("\n=== 規模区分（ScaleCat） ===")
    scale = sym_pnl.groupby("scale").agg(
        n=("group", "count"),
        win_rate=("group", lambda s: (s == "winner").mean()),
        avg_pnl=("total_pnl", "mean"),
    )
    print(scale.to_string())

    out = Path(CONFIG["output_dir"])
    sym_pnl.to_csv(out / "symbol_diag_bb_ma60.csv", encoding="utf-8-sig")
    cmp.to_csv(out / "winner_loser_compare.csv", encoding="utf-8-sig", index=False)
    print(f"\n[INFO] 保存: {out / 'symbol_diag_bb_ma60.csv'}")


if __name__ == "__main__":
    main()