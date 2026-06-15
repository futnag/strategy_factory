"""C2（大量保有アクティビスト新規参入）後ドリフトの記述統計。

母集団＝large_holdings.parquet の report_class=='new' & is_important_proposal（docs/10 §7）。
買収 SPV 偽陽性を保有割合バンドで除外（既定 5–30%・docs/10 §8.3）。エントリーは **提出日 T+1
始値**（PIT・docs/10 §5）、前方リターンは AdjO(T+1)→AdjC(T+1+h)。TOPIX(code_0000) 超過も併記。

**確定事項 3**：これは厳格 DSR 裁定ではなく記述統計＋Phase 2 フォワードの素材。閾値最適化・
事後レジーム選択はしない（K 規律）。delisting（TOB 成立等）で長期窓が欠ける選択バイアスは
カバレッジとして明示する。

usage:
  python examples/research_c2_drift.py
  python examples/research_c2_drift.py --min-ratio 0.05 --max-ratio 0.30
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EDINET_LIST = "data/edinet/list/*.parquet"
LVH = "data/edinet/large_holdings.parquet"
JQ_DAILY = "data/jquants/daily"
TOPIX = "data/jquants/indices/code_0000.parquet"
HORIZONS = [5, 20, 60, 120]            # 取引日（≈1週/1月/3月/6月）


def edinet_to_sec() -> pd.Series:
    """EDINET 一覧ミラーから edinetCode→secCode（対象会社自身の提出に secCode が載る）。"""
    frames = []
    for p in sorted(glob.glob(EDINET_LIST)):
        d = pd.read_parquet(p)
        if "_empty" not in d.columns:
            frames.append(d[["edinetCode", "secCode"]])
    m = pd.concat(frames, ignore_index=True).dropna()
    m = m[m["secCode"].astype(str).str.len() >= 4]
    return m.drop_duplicates("edinetCode").set_index("edinetCode")["secCode"]


def load_universe(min_ratio: float, max_ratio: float) -> pd.DataFrame:
    df = pd.read_parquet(LVH)
    u = df[(df["report_class"] == "new") & (df["is_important_proposal"] == True)].copy()  # noqa: E712
    u = u[(u["holding_ratio"] >= min_ratio) & (u["holding_ratio"] < max_ratio)].copy()
    u["sec"] = u["issuer_edinet"].map(edinet_to_sec())
    u = u.dropna(subset=["sec"]).copy()
    u["submit_dt"] = pd.to_datetime(u["submit_dt"])
    u["entry_date"] = u["submit_dt"].dt.normalize()
    u["year"] = u["submit_dt"].dt.year
    u["style_grp"] = u["style"].fillna("untagged")
    bins = [0, 0.05, 0.10, 0.20, 0.30, 1.01]
    u["band"] = pd.cut(u["holding_ratio"], bins=bins,
                       labels=["<5%", "5-10%", "10-20%", "20-30%", ">=30%"], right=False)
    return u


def price_panel(codes: set[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """対象コードの調整後始値/終値の時系列（by-date ミラーを範囲読み）。"""
    files = []
    for p in sorted(glob.glob(f"{JQ_DAILY}/*.parquet")):
        ymd = Path(p).stem
        if start <= ymd <= end:
            files.append(p)
    rows = []
    for p in files:
        d = pd.read_parquet(p)
        if "Code" not in d.columns or d.empty:
            continue
        o = "AdjO" if "AdjO" in d.columns else ("O" if "O" in d.columns else None)
        c = "AdjC" if "AdjC" in d.columns else ("C" if "C" in d.columns else None)
        if o is None or c is None:
            continue
        sub = d[d["Code"].isin(codes)][["Date", "Code", o, c]].copy()
        sub.columns = ["Date", "Code", "AdjO", "AdjC"]
        rows.append(sub)
    panel = pd.concat(rows, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    out = {}
    for code, g in panel.groupby("Code"):
        out[code] = g.sort_values("Date").set_index("Date")[["AdjO", "AdjC"]]
    return out


def topix_series() -> pd.DataFrame:
    t = pd.read_parquet(TOPIX)
    t["Date"] = pd.to_datetime(t["Date"])
    return t.sort_values("Date").set_index("Date")[["O", "C"]]


def forward_returns(u: pd.DataFrame, panel: dict, topix: pd.DataFrame) -> pd.DataFrame:
    """各イベントの T+1 始値エントリー → 各ホライズンの素・TOPIX 超過リターン。"""
    recs = []
    tdates = topix.index
    for _, r in u.iterrows():
        s = panel.get(r["sec"])
        rec = {"sec": r["sec"], "entry_date": r["entry_date"], "year": r["year"],
               "style_grp": r["style_grp"], "band": r["band"],
               "canonical_group": r["canonical_group"]}
        if s is None or s.empty:
            recs.append(rec)
            continue
        fwd = s.index[s.index > r["entry_date"]]          # T+1 = 提出日翌取引日
        if len(fwd) == 0:
            recs.append(rec)
            continue
        e = fwd[0]
        entry = s.at[e, "AdjO"]
        epos = s.index.get_loc(e)
        # TOPIX 同期間（エントリー日始値→各日終値）
        tpos = tdates.get_indexer([e], method="nearest")[0]
        t_entry = topix.iloc[tpos]["O"]
        for h in HORIZONS:
            if epos + h < len(s) and entry and entry > 0:
                rec[f"r{h}"] = s.iloc[epos + h]["AdjC"] / entry - 1.0
                if tpos + h < len(topix) and t_entry > 0:
                    tr = topix.iloc[tpos + h]["C"] / t_entry - 1.0
                    rec[f"x{h}"] = rec[f"r{h}"] - tr
        recs.append(rec)
    return pd.DataFrame(recs)


def _agg(df: pd.DataFrame, label: str) -> dict:
    out = {"group": label, "N": len(df)}
    for h in HORIZONS:
        col = f"r{h}"
        v = df[col].dropna() if col in df else pd.Series(dtype=float)
        x = df[f"x{h}"].dropna() if f"x{h}" in df else pd.Series(dtype=float)
        out[f"n{h}"] = len(v)
        out[f"med_r{h}"] = round(v.median() * 100, 2) if len(v) else np.nan
        out[f"med_x{h}"] = round(x.median() * 100, 2) if len(x) else np.nan
        out[f"mean_x{h}"] = round(x.mean() * 100, 2) if len(x) else np.nan
        out[f"hit{h}"] = round((x > 0).mean() * 100, 1) if len(x) else np.nan
    return out


def report(fr: pd.DataFrame) -> str:
    lines = []

    def table(title, groups):
        lines.append(f"\n### {title}")
        hdr = ("| 群 | N | " + " | ".join(
            [f"中央超過x{h}% (平均{h}/勝率{h}/n)" for h in HORIZONS]) + " |")
        lines.append(hdr)
        lines.append("|" + "---|" * (2 + len(HORIZONS)))
        for lab, sub in groups:
            a = _agg(sub, lab)
            cells = " | ".join(
                [f"{a[f'med_x{h}']} ({a[f'mean_x{h}']}/{a[f'hit{h}']}%/{a[f'n{h}']})"
                 for h in HORIZONS])
            lines.append(f"| {lab} | {a['N']} | {cells} |")

    table("全体", [("all", fr)])
    table("スタイル別", [(g, fr[fr["style_grp"] == g])
                        for g in ["engagement_hard", "engagement_soft", "untagged"]])
    table("保有割合帯別", [(b, fr[fr["band"] == b])
                          for b in ["5-10%", "10-20%", "20-30%"] if (fr["band"] == b).any()])
    table("参入年別", [(str(y), fr[fr["year"] == y])
                      for y in sorted(fr["year"].dropna().unique())])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ratio", type=float, default=0.05)
    ap.add_argument("--max-ratio", type=float, default=0.30)
    args = ap.parse_args()

    u = load_universe(args.min_ratio, args.max_ratio)
    print(f"母集団（保有割合 {args.min_ratio:.0%}–{args.max_ratio:.0%}・sec 解決済）"
          f"：{len(u)} 件 / 対象 {u['sec'].nunique()} 社 / 提出者 {u['filer_edinet'].nunique()}")
    start = (u["entry_date"].min() - pd.Timedelta(days=10)).strftime("%Y%m%d")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    panel = price_panel(set(u["sec"]), start, end)
    fr = forward_returns(u, panel, topix_series())
    cov = {h: int(fr[f"r{h}"].notna().sum()) for h in HORIZONS if f"r{h}" in fr}
    print(f"前方窓カバレッジ（delisting/直近で欠落）：{cov}")
    print(report(fr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
