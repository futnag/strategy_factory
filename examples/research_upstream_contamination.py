"""P1-B: 特徴量計算の上流汚染監査 — docs/04 P1-B・C3(arXiv:2507.07107)。

C3 の指摘：値幅制限で**約定不能な価格**がローリング・ファクター計算に素通しで混入する
だけで、見かけ IC を水増しし実現 Sharpe を毀損する（A 株実データ・mask-first 設計で解決）。
日本にも値幅制限（ストップ高/安の引け張り付き）・出来高ゼロ日が存在し問題は同型。

本スクリプトは監査と影響の定量化のみ（コード変更なし・建玉なし・K 消費ゼロ）：
 1. 汚染頻度：張り付き（`frictions.limit_lock_flags`＝引け張り付き）・出来高ゼロの
    セル（銘柄×日）頻度を全期間で集計（全パネル / PIT 上位300 ユニバース内の両方）。
 2. 影響の定量化：Gold 層の価格特徴量（momentum_12_1 / reversal_5 / vol_20）について
    mask-first（非約定可能日の価格を NaN）あり/なしの特徴量差分と、月次リバランスの
    上下クインタイル（20%・旗艦と同分位）構成の入替率を測る。
 3. value（B/M）の直接汚染：月末リバランス時点の生株価が張り付き値であるセルの頻度
    （B/M の分母に約定不能価格が直接入るケース）。

実行: .venv\\Scripts\\python.exe examples\\research_upstream_contamination.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.data.store import load_wide  # noqa: E402
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    filter_common_stocks, point_in_time_universe, universe_members,
)

START, END = "2016-07", "2026-05"
Q = 0.2                      # 旗艦と同じ上下クインタイル
MOM_SKIP, MOM_LB = 21, 252   # Gold 層 momentum_12_1 と同一
REV_W, VOL_W = 5, 20


def _quintile_turnover(raw_row: pd.Series, masked_row: pd.Series, q: float
                       ) -> tuple[float, float] | None:
    """1ヶ月断面の上下クインタイル構成の入替率（mask 適用前後）。"""
    a = raw_row.dropna()
    b = masked_row.dropna()
    if len(a) < 25 or len(b) < 25:
        return None
    k = max(1, int(len(a) * q))
    top_a = set(a.sort_values().index[-k:])
    bot_a = set(a.sort_values().index[:k])
    kb = max(1, int(len(b) * q))
    top_b = set(b.sort_values().index[-kb:])
    bot_b = set(b.sort_values().index[:kb])
    return (1.0 - len(top_a & top_b) / len(top_a),
            1.0 - len(bot_a & bot_b) / len(bot_a))


def main() -> int:
    # --- 日次パネル（Silver）と約定可能性フラグ ---
    print("日次パネル読込中（Silver）...")
    close = load_wide("close", start="2016-06")
    high = load_wide("high", start="2016-06")
    low = load_wide("low", start="2016-06")
    ul = load_wide("upper_limit", start="2016-06")
    ll = load_wide("lower_limit", start="2016-06")
    vo = load_wide("volume", start="2016-06")
    adj = load_wide("adj_close", start="2016-06")
    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll, volume=vo)
    pinned = no_buy | no_sell                  # 引け張り付き（売買いずれか不能）
    vol0 = vo.reindex_like(close) == 0
    non_tradable = pinned                      # limit_lock_flags が vol0 を両側に含む
    has_px = close.notna()

    n_cells = int(has_px.to_numpy().sum())
    n_pin = int((pinned & has_px).to_numpy().sum())
    n_v0 = int((vol0 & has_px).to_numpy().sum())
    print(f"\n=== 1) 汚染頻度（全パネル {close.shape[0]}日 × {close.shape[1]}銘柄・"
          f"{START}〜{END}）===")
    print(f"  価格セル総数            : {n_cells:,}")
    print(f"  引け張り付き（UL/LL）   : {n_pin:,}（{n_pin / n_cells:.4%}）")
    print(f"  出来高ゼロ              : {n_v0:,}（{n_v0 / n_cells:.4%}）")
    per_year = (pinned & has_px).groupby(pinned.index.year).sum().sum(axis=1)
    print("  年別の張り付きセル数    : "
          + " ".join(f"{y}:{int(c):,}" for y, c in per_year.items()))

    # --- PIT 上位300 ユニバース（旗艦と同一構築）---
    print("\nPIT ユニバース構築中（月次・旗艦と同一）...")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    turn_m = assemble_panel(snaps, "Va")
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn_m[[c for c in turn_m.columns if str(c) in common]]
    umask_m = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask_m)
    rebal = turn_m.index

    # 月次マスクを日次へ展開（各日はその月のメンバーシップ・ffill）
    cols = [c for c in superset if c in close.columns]
    umask_d = (umask_m.reindex(columns=cols).fillna(False)
               .reindex(close.index.union(umask_m.index)).ffill()
               .reindex(close.index).fillna(False).astype(bool))
    in_u = umask_d & has_px[cols]
    n_u = int(in_u.to_numpy().sum())
    n_u_pin = int((pinned[cols] & in_u).to_numpy().sum())
    print(f"=== 1b) 汚染頻度（PIT 上位300 ユニバース内・日次）===")
    print(f"  ユニバース内セル        : {n_u:,}")
    print(f"  うち引け張り付き        : {n_u_pin:,}（{n_u_pin / n_u:.4%}）")

    # --- 2) mask-first あり/なしの特徴量差分と分位入替（ユニバース・月次断面）---
    print("\n=== 2) mask-first の影響（Gold 価格特徴量・月末断面・ユニバース内）===")
    adj_m = adj.where(~non_tradable.reindex_like(adj).fillna(False))
    feats = {
        "momentum_12_1": (adj.shift(MOM_SKIP) / adj.shift(MOM_LB) - 1.0,
                          adj_m.shift(MOM_SKIP) / adj_m.shift(MOM_LB) - 1.0),
        f"reversal_{REV_W}": (-(adj / adj.shift(REV_W) - 1.0),
                              -(adj_m / adj_m.shift(REV_W) - 1.0)),
        f"vol_{VOL_W}": (
            adj.pct_change(fill_method=None).rolling(VOL_W, min_periods=10).std(),
            adj_m.pct_change(fill_method=None).rolling(VOL_W, min_periods=10).std()),
    }
    month_ends = [t for t in rebal if t in adj.index]
    print(f"  月末断面 {len(month_ends)}本 × ユニバース約300銘柄、クインタイル q={Q:g}")
    print(f"  {'feature':<15} {'対象セル':>9} {'NaN化':>8} {'値変化':>8} "
          f"{'top20%入替':>10} {'bot20%入替':>10}")
    for name, (f_raw, f_msk) in feats.items():
        tot = nanified = changed = 0
        top_t, bot_t = [], []
        for t in month_ends:
            m = umask_d.loc[t]
            members = m[m].index
            a = f_raw.loc[t].reindex(members)
            b = f_msk.loc[t].reindex(members)
            ok = a.notna()
            tot += int(ok.sum())
            nanified += int((ok & b.isna()).sum())
            both = ok & b.notna()
            changed += int((both & ((a - b).abs() > 1e-12)).sum())
            qt = _quintile_turnover(a, b, Q)
            if qt is not None:
                top_t.append(qt[0])
                bot_t.append(qt[1])
        print(f"  {name:<15} {tot:>9,} {nanified / tot:>8.4%} {changed / tot:>8.4%} "
              f"{np.mean(top_t):>10.4%} {np.mean(bot_t):>10.4%}")

    # --- 3) value（B/M）の直接汚染：月末の生株価が張り付き値 ---
    print("\n=== 3) value（B/M 分母）の直接汚染（月末リバランス時点・ユニバース内）===")
    hi_m, lo_m = (assemble_panel(snaps, c) for c in ("H", "L"))
    ul_m, ll_m = (assemble_panel(snaps, c) for c in ("UL", "LL"))
    vo_m = assemble_panel(snaps, "Vo")
    raw_m = assemble_panel(snaps, "C")
    nb_m, ns_m = limit_lock_flags(raw_m, hi_m, lo_m, ul_m, ll_m, volume=vo_m)
    pin_m = (nb_m | ns_m).reindex(index=rebal, columns=cols).fillna(False)
    um = umask_m.reindex(columns=cols).fillna(False)
    n_um = int((um & raw_m.reindex(index=rebal, columns=cols).notna()).to_numpy().sum())
    n_um_pin = int((pin_m & um).to_numpy().sum())
    print(f"  月末×ユニバースのセル   : {n_um:,}")
    print(f"  うち月末終値が張り付き  : {n_um_pin:,}（{n_um_pin / n_um:.4%}）"
          f" ＝ B/M がその値幅制限価格で計算される頻度")

    print("\n※ 読み方：C3 の警告が効くのは『張り付き価格がローリング統計と分位構成を"
          "実質的に動かす』場合。NaN化・値変化・分位入替がいずれも ~0.1% 未満なら、"
          "月次・流動性上位300・終値リバランスの本リポジトリ構成では上流汚染は実害なし"
          "（§6.12 の執行面の結論と同型）。日次・小型・イベント系を裁く際は mask-first を"
          "前提とすること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
