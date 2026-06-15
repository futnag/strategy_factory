"""C1（TOB リスクアーブ）スプレッド×不成立 診断と K=8 格子 P&L（docs/09 §3-4）。

M&A Online TOB 表（2016+・現金・上場対象）をスパインに、EDINET tob_deals の当初/最終価格で
`_displayed` リークを根治（tob_arb.deal_metrics）。スプレッドと成否のクロス集計で「高スプレッド
＝不成立セレクション」を診断し、事前登録格子（サブ期間×競合×スプレッド閾値）の各セルで
等加重 P&L を出す。**これは prior を固める診断**＝scope `tob_arb` の正式 DSR 裁定は別段
（judge_grid 統合は次段・確定事項/K 規律）。

usage:
  python examples/research_c1_tob_arb.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.equities import tob_arb as ta          # noqa: E402

DETAIL = "data/manual/maonline/parsed/tob_detail.csv"
DEALS = "data/edinet/tob_deals.parquet"
JQ_DAILY = "data/jquants/daily"


def jq_code(code4: Optional[str]) -> Optional[str]:
    if not code4 or pd.isna(code4):
        return None
    s = str(code4).strip()
    return s + "0" if len(s) == 4 else s


def load_spine() -> pd.DataFrame:
    d = pd.read_csv(DETAIL, dtype=str)
    d["announce_date"] = pd.to_datetime(d["announce_date"], errors="coerce")
    d["year"] = d["announce_date"].dt.year
    d["price_displayed"] = pd.to_numeric(d["price_displayed_jpy"], errors="coerce")
    d = d[d["target_code"].notna() & (d["target_code"].str.len() >= 4)]   # 上場対象
    d = d[d["price_displayed"].notna()]                                   # 現金（円価格）代理
    d = d[(d["year"] >= 2016) & (d["announce_date"].notna())].copy()
    d["sec"] = d["target_code"].map(jq_code)
    d["competing"] = ta.mark_competing(d)
    d["subperiod"] = d["year"].map(ta.subperiod)
    return d


def join_edinet(d: pd.DataFrame) -> pd.DataFrame:
    """EDINET 当初/最終価格・バンプ数を target_edinet＋公表日近接で突合（2021+ を根治）。"""
    e = pd.read_parquet(DEALS)
    e["announce_dt"] = pd.to_datetime(e["announce_dt"], errors="coerce")
    e = e.dropna(subset=["target_edinet"])
    d["initial_price"] = d["price_displayed"]              # 既定＝表示価格（当初の代理）
    d["final_price"] = d["price_displayed"]
    d["price_source"] = "displayed"
    d["n_bumps"] = 0
    by_edinet = {k: g for k, g in e.groupby("target_edinet")}
    for i, r in d.iterrows():
        ec = r.get("target_edinet_code")
        if not ec or ec not in by_edinet:
            continue
        g = by_edinet[ec]
        dt = (g["announce_dt"] - r["announce_date"]).abs()
        j = g.loc[dt.idxmin()]
        if dt.min() > pd.Timedelta(days=45) or pd.isna(j.get("initial_price")):
            continue
        ip = float(j["initial_price"])
        disp = float(r["price_displayed"])
        # 整合：EDINET 抽出価格が表示(最終)価格から大きく乖離する案件は普通株でなく
        # 新株予約権等の単価を拾った抽出エラー（例 ペイロール 87000 vs 実 ~1310）。バンプは
        # 通常 <30% なので、相対乖離 >50% は誤抽出とみなし表示価格に戻す（リークより誤値が害）。
        if disp > 0 and abs(ip - disp) / disp > 0.50:
            d.at[i, "price_source"] = "edinet_rejected"
            continue
        d.at[i, "initial_price"] = ip
        d.at[i, "final_price"] = float(j["final_price"]) if pd.notna(
            j.get("final_price")) else ip
        d.at[i, "price_source"] = "edinet"
        d.at[i, "n_bumps"] = int(j.get("n_bumps") or 0)
    return d


def price_panel(codes: set, start: str, end: str) -> dict:
    out: dict = {}
    rows = []
    cols = ["Date", "Code", "O", "H", "L", "C", "UL"]
    for p in sorted(glob.glob(f"{JQ_DAILY}/*.parquet")):
        ymd = Path(p).stem
        if not (start <= ymd <= end):
            continue
        df = pd.read_parquet(p)
        if "Code" not in df.columns or "O" not in df.columns or df.empty:
            continue
        have = [c for c in cols if c in df.columns]
        rows.append(df[df["Code"].isin(codes)][have])
    if not rows:
        return out
    panel = pd.concat(rows, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    keep = [c for c in ["O", "H", "L", "C", "UL"] if c in panel.columns]
    for code, g in panel.groupby("Code"):
        out[code] = g.sort_values("Date").set_index("Date")[keep]
    return out


def _locked_limit_up(bar) -> bool:
    """ストップ高ロック＝約定不能の判定（UL=1 かつ高安同値＝終日単一値で売り無し）。"""
    try:
        return (float(bar.get("UL", 0)) == 1
                and pd.notna(bar.get("H")) and pd.notna(bar.get("L"))
                and float(bar["H"]) == float(bar["L"]))
    except Exception:                                       # noqa: BLE001
        return False


def compute(d: pd.DataFrame, panel: dict) -> pd.DataFrame:
    recs = []
    for _, r in d.iterrows():
        s = panel.get(r["sec"])
        rec = {"year": r["year"], "subperiod": r["subperiod"],
               "competing": r["competing"], "result": r["tob_result"],
               "price_source": r["price_source"], "n_bumps": r["n_bumps"],
               "arb_spread": None, "ret": None, "premium": None, "fillable": True}
        if s is None or s.empty:
            recs.append(rec)
            continue
        anchor = r["announce_date"].normalize()
        before = s.index[s.index < anchor]
        after = s.index[s.index > anchor]
        prev_close = float(s.at[before[-1], "C"]) if len(before) else None
        entry_open = float(s.at[after[0], "O"]) if len(after) else None
        if len(after):
            rec["fillable"] = not _locked_limit_up(s.loc[after[0]])
        end = pd.to_datetime(r.get("end_date_displayed"), errors="coerce")
        exit_mkt = None
        if pd.notna(end):
            onafter = s.index[s.index >= end.normalize()]
            if len(onafter):
                exit_mkt = float(s.at[onafter[0], "C"])
        m = ta.deal_metrics(r["tob_result"], r["initial_price"], r["final_price"],
                            prev_close, entry_open, exit_mkt)
        rec.update(m)
        recs.append(rec)
    return pd.DataFrame(recs)


def main() -> int:
    d = load_spine()
    d = join_edinet(d)
    print(f"スパイン（2016+・上場・現金代理）：{len(d)} 件 / "
          f"EDINET 当初価格突合 {int((d['price_source']=='edinet').sum())} 件 / "
          f"バンプ有 {int((d['n_bumps']>0).sum())} 件")
    start = (d["announce_date"].min() - pd.Timedelta(days=10)).strftime("%Y%m%d")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    panel = price_panel(set(d["sec"].dropna()), start, end)
    m = compute(d, panel)
    ok = m.dropna(subset=["arb_spread"])
    rd_all = ok[ok["result"].isin(["成立", "不成立"])]
    locked = int((~rd_all["fillable"]).sum())
    rd = rd_all[rd_all["fillable"]].copy()                  # 約定可能エントリーのみ
    print(f"株価照合 {len(ok)} 件 / 成否確定 {len(rd_all)} 件 / "
          f"不成立 {(rd_all['result']=='不成立').sum()} 件")
    print(f"  ストップ高ロック（T+1 約定不能・DP15）で除外：{locked} 件 → 約定可能 {len(rd)} 件")

    print("\n--- 不成立率・サブ期間（レジーム分離・docs/09 §2.1・約定可能のみ） ---")
    for sp in ["2016-2019", "2020-2026"]:
        g = rd[rd["subperiod"] == sp]
        if len(g):
            print(f"  {sp}: n={len(g):3d}  不成立率 {(g['result']=='不成立').mean():.1%}")

    print("\n--- スプレッド×成否（高スプレッド帯のセレクション・docs/09 §3 prior 検証） ---")
    print("  （ロック前 all / ロック除外 fillable を併記＝幻のエッジの所在を可視化）")
    rb_all = rd_all.copy(); rb_all["bucket"] = rb_all["arb_spread"].map(ta.spread_bucket)
    rb = rd.copy(); rb["bucket"] = rb["arb_spread"].map(ta.spread_bucket)
    for b in ["<1%", "1-3%", "3-7%", ">=7%"]:
        ga, gf = rb_all[rb_all["bucket"] == b], rb[rb["bucket"] == b]
        if len(ga):
            print(f"  spread {b:>5}: all n={len(ga):3d} 不成立{(ga['result']=='不成立').mean():4.0%} "
                  f"平均ret{ga['ret'].dropna().mean():+7.2%} | fillable n={len(gf):3d} "
                  f"不成立{(gf['result']=='不成立').mean() if len(gf) else float('nan'):4.0%} "
                  f"平均ret{gf['ret'].dropna().mean() if len(gf) else float('nan'):+7.2%} "
                  f"中央{gf['ret'].dropna().median() if len(gf) else float('nan'):+7.2%}")

    print("\n--- K=8 格子（サブ期間×競合×スプレッド閾値・約定可能・等加重 P&L 診断） ---")
    print(f"  {'cell':32s} {'N':>4} {'不成立%':>7} {'平均ret':>8} {'中央ret':>8} {'勝率':>6}")
    for sp in ["2016-2019", "2020-2026"]:
        for comp_label, comp_vals in [("競合除外", [False]), ("競合含む", [True, False])]:
            for thr in [0.01, 0.03]:
                g = rd[(rd["subperiod"] == sp) & (rd["competing"].isin(comp_vals))
                       & (rd["arb_spread"] >= thr)]
                rr = g["ret"].dropna()
                if not len(rr):
                    continue
                cell = f"{sp}/{comp_label}/spread>={int(thr*100)}%"
                print(f"  {cell:32s} {len(rr):4d} {(g['result']=='不成立').mean():6.1%} "
                      f"{rr.mean():+7.2%} {rr.median():+7.2%} {(rr>0).mean():5.1%}")
    print("\n注：診断（prior 確認）。正式 scope tob_arb の DSR 裁定は judge_grid 統合後（K 規律）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
