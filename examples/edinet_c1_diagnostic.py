"""C1（TOB リスクアーブ）の Phase A 診断（EDINET 2022+・docs/09 §4）。

enrich 済み案件テーブル（data/edinet/tob_deals.parquet）＋ J-Quants 株価で、不成立率・
スプレッド分布・スプレッド×成否のセレクション・保有期間・簡易 P&L を診断する。

重要な限界（正直に）：
- EDINET 縦覧の都合で窓は 2022+ のみ＝**post-2020 本格化レジーム単一**。docs/09 §2.1 の
  2016–2019 / 2020+ サブ期間分割は M&A Online 履歴が要る（本診断ではできない）。
- PIT アンカーは 240 届出日（M&A Online の公表日より遅い側＝保守的。事前公表型は公表日が
  ~2 週早く、ここで測るスプレッドは届出後の狭い側）。エントリは届出日 T+1 始値。
- これは prior を整える**診断**であって scope 裁定ではない（DSR・事前登録は docs/09 §4
  step4＝M&A Online 投入後）。現金/株式交換の別は EDINET 単独で機械判定できないため未分離。

usage:
  python examples/edinet_c1_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import edinet as ed          # noqa: E402
from invest_system.data.sources import jquants as jq          # noqa: E402

DEALS = ed._CACHE / "tob_deals.parquet"


def _prices(sec: str) -> pd.DataFrame | None:
    """対象銘柄の日次株価（生 O/C）。J-Quants キャッシュ付き。失敗は None。"""
    try:
        df = jq.fetch_daily_history(str(sec))
    except Exception:                                   # noqa: BLE001
        return None
    if df.empty or "Date" not in df.columns:
        return None
    return df.sort_values("Date").reset_index(drop=True)


def _prev_close(px: pd.DataFrame, anchor: pd.Timestamp):
    b = px[px["Date"] < anchor]
    return float(b["C"].iloc[-1]) if len(b) and pd.notna(b["C"].iloc[-1]) else None


def _next_open(px: pd.DataFrame, anchor: pd.Timestamp):
    a = px[px["Date"] > anchor]
    if not len(a):
        return None
    v = a["O"].iloc[0]
    return float(v) if pd.notna(v) else None


def _close_on_after(px: pd.DataFrame, date: pd.Timestamp):
    a = px[px["Date"] >= date]
    return float(a["C"].iloc[0]) if len(a) and pd.notna(a["C"].iloc[0]) else None


def build_metrics(deals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, d in deals.iterrows():
        sec, ip = d.get("target_sec"), d.get("initial_price")
        if pd.isna(sec) or pd.isna(ip):
            continue
        anchor = pd.Timestamp(d["announce_dt"]).normalize()
        px = _prices(sec)
        if px is None:
            continue
        prev_c = _prev_close(px, anchor)
        entry = _next_open(px, anchor)
        if not prev_c or not entry:
            continue
        end = pd.to_datetime(d.get("period_end_final"), errors="coerce")
        exit_mkt = _close_on_after(px, end) if pd.notna(end) else None
        result = d.get("result")
        fp = d.get("final_price")
        fp = float(fp) if pd.notna(fp) else float(ip)
        # 簡易 P&L：T+1 始値で建て、成立=買付価格で出口 / 不成立=実終了日近傍の市場価格
        ret = None
        if result == "成立":
            ret = fp / entry - 1.0
        elif result == "不成立" and exit_mkt:
            ret = exit_mkt / entry - 1.0
        rows.append({
            "deal_id": d["deal_id"], "year": anchor.year, "target_sec": sec,
            "result": result, "competing": d.get("competing"),
            "premium": ip / prev_c - 1.0,           # 公表前日終値に対する買付プレミアム
            "arb_spread": ip / entry - 1.0,         # T+1始値で買い→買付価格までの裁定幅
            "holding_days": (end - anchor).days if pd.notna(end) else np.nan,
            "ret": ret,
        })
    return pd.DataFrame(rows)


def main() -> int:
    if not DEALS.exists():
        raise SystemExit("tob_deals.parquet が無い。先に edinet_tob_backfill.py を実行。")
    deals = pd.read_parquet(DEALS)
    deals = deals[pd.to_datetime(deals["announce_dt"]).dt.year >= 2022]
    print(f"=== C1 Phase A 診断（EDINET 2022+・{len(deals)} 案件） ===")
    print("※ 単一レジーム（post-2020）・届出日アンカー・診断であって scope 裁定ではない\n")

    m = build_metrics(deals)
    print(f"株価照合できた案件: {len(m)} / 成否確定: {m['result'].notna().sum()}")

    print("\n--- 不成立率（年別・成否確定分） ---")
    rd = m[m["result"].notna()]
    for y in sorted(rd["year"].unique()):
        g = rd[rd["year"] == y]
        fr = (g["result"] == "不成立").mean()
        print(f"  {y}: n={len(g):3d}  不成立率 {fr:.1%}")
    print(f"  全体: n={len(rd)}  不成立率 {(rd['result']=='不成立').mean():.1%}  "
          f"競合 {int(rd['competing'].sum())} 件")

    print("\n--- 保有期間（届出→実終了・暦日） ---")
    hd = m["holding_days"].dropna()
    if len(hd):
        print(f"  中央値 {hd.median():.0f}日 / 平均 {hd.mean():.0f}日 / "
              f"IQR[{hd.quantile(.25):.0f}, {hd.quantile(.75):.0f}]")

    print("\n--- 裁定スプレッド（T+1始値→買付価格） ---")
    sp = m["arb_spread"].dropna()
    print(f"  中央値 {sp.median():.2%} / 平均 {sp.mean():.2%} / "
          f"p10 {sp.quantile(.10):.2%} / p90 {sp.quantile(.90):.2%}")
    print(f"  公表プレミアム中央値 {m['premium'].median():.1%}")

    print("\n--- スプレッド×成否（高スプレッド帯のセレクション検証・docs/09 §3） ---")
    rb = rd.dropna(subset=["arb_spread"]).copy()
    rb["bucket"] = pd.cut(rb["arb_spread"], [-1, 0.01, 0.03, 0.07, 1.0],
                          labels=["<1%", "1-3%", "3-7%", ">7%"])
    for b, g in rb.groupby("bucket", observed=True):
        print(f"  spread {str(b):>5}: n={len(g):3d}  不成立率 {(g['result']=='不成立').mean():.1%}  "
              f"平均リターン {g['ret'].dropna().mean():+.2%}")

    print("\n--- 簡易 P&L（診断：T+1始値→成立=買付価格/不成立=市場・等加重） ---")
    rr = m["ret"].dropna()
    if len(rr):
        print(f"  n={len(rr)}  平均 {rr.mean():+.2%} / 中央値 {rr.median():+.2%} / "
              f"勝率 {(rr > 0).mean():.1%}")
        print(f"  成立時 平均 {m.loc[m['result']=='成立','ret'].mean():+.2%} / "
              f"不成立時 平均 {m.loc[m['result']=='不成立','ret'].dropna().mean():+.2%}")
    print("\n注：DSR/事前登録は未実施（診断段階）。長期履歴・公表日は M&A Online 投入後。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
