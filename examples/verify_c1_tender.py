"""C1 テンダー執行前提の検証：按分（proration）がリスクアーブ収益を過大評価していないか。

正式裁定（docs/09 §6）の正のリターンは「成立時に保有株すべてを買付価格でテンダーして受け取る」
前提に依存する。最大の穴は **部分買付（上限あり）の按分比例**：上限ありTOBが応募超過になると
一部しか買付価格で買われず、残りは TOB 後の市場で処分（通常下落）。

検証：EDINET 公開買付報告書(270) から「応募数 vs 買付数（株券）」を抽出し按分率を実測。
各成立案件のリターンを (a) フルテンダー前提と (b) 按分補正
  [買付価格 × 按分率 + 終了後市場価格 × (1−按分率)]
で計算し、スプレッド帯別・全体の差と Sharpe への影響を出す。

usage:
  python examples/verify_c1_tender.py
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import edinet as ed          # noqa: E402
from invest_system.equities import tob_arb as ta             # noqa: E402
from invest_system.equities import tob_events as te          # noqa: E402

JQ_DAILY = "data/jquants/daily"
FAIL_BUFFER = 5
_Z2H = str.maketrans("０１２３４５６７８９，", "0123456789,")
_PROR_RE = re.compile(r"株券([0-9,]+)[（(]株[)）]([0-9,]+)[（(]株[)）]")


def load_listing() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(glob.glob("data/edinet/list/*.parquet"))]
    return pd.concat([f for f in frames if "_empty" not in f.columns], ignore_index=True)


def proration_ratio(deal_id: str, tob: pd.DataFrame) -> float:
    """270 の株券『応募数/買付数』から按分率 acquired/tendered（取得不能/全部買付=1.0）。"""
    reps = tob[(tob["parentDocID"] == deal_id) & (tob["docTypeCode"] == "270")]
    if not len(reps):
        return 1.0
    try:
        raw = te.doc_csv_records(reps.iloc[-1]["docID"]).get(te.E_ACQUIRED) or ""
    except Exception:                                          # noqa: BLE001
        return 1.0
    m = _PROR_RE.search(raw.translate(_Z2H).replace(" ", ""))
    if not m:
        return 1.0
    tend, acq = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    return acq / tend if tend > 0 else 1.0


def price_panel(codes: set, start: str) -> dict:
    rows = []
    for p in sorted(glob.glob(f"{JQ_DAILY}/*.parquet")):
        if Path(p).stem < start:
            continue
        df = pd.read_parquet(p)
        if "Code" not in df.columns or "O" not in df.columns or df.empty:
            continue
        rows.append(df[df["Code"].isin(codes)][["Date", "Code", "O", "C"]])
    long = pd.concat(rows, ignore_index=True)
    long["Date"] = pd.to_datetime(long["Date"])
    return {c: g.sort_values("Date").set_index("Date")[["O", "C"]]
            for c, g in long.groupby("Code")}


def main() -> int:
    listing = load_listing()
    tob = listing[listing["ordinanceCode"] == "040"]
    deals = pd.read_parquet("data/edinet/tob_deals.parquet")
    deals = deals[deals["result"] == "成立"].copy()
    deals["announce_dt"] = pd.to_datetime(deals["announce_dt"])
    deals = deals.dropna(subset=["target_sec", "initial_price", "final_price"])
    deals["sec"] = deals["target_sec"].astype(str)
    panel = price_panel(set(deals["sec"]),
                        (deals["announce_dt"].min() - pd.Timedelta(days=10)).strftime("%Y%m%d"))

    rows = []
    for _, d in deals.iterrows():
        s = panel.get(d["sec"])
        if s is None or s.empty:
            continue
        after = s.index[s.index > d["announce_dt"].normalize()]
        if len(after) == 0:
            continue
        entry = float(s.at[after[0], "O"])
        if entry <= 0:
            continue
        end = pd.to_datetime(d.get("period_end_final"), errors="coerce")
        onafter = s.index[s.index >= end.normalize()] if pd.notna(end) else after[:0]
        # 終了後の処分価格：終了日＋バッファの終値（按分残の市場処分）
        post = None
        if len(onafter):
            pos = min(s.index.get_loc(onafter[0]) + FAIL_BUFFER, len(s) - 1)
            post = float(s.iloc[pos]["C"])
        pror = proration_ratio(d["deal_id"], tob)
        spread = float(d["initial_price"]) / entry - 1.0
        ret_full = float(d["final_price"]) / entry - 1.0
        if post is not None:
            ret_corr = (d["final_price"] * pror + post * (1 - pror)) / entry - 1.0
        else:
            ret_corr = ret_full if pror > 0.999 else np.nan
        rows.append({"sec": d["sec"], "spread": spread, "pror": pror,
                     "ret_full": ret_full, "ret_corr": ret_corr,
                     "bucket": ta.spread_bucket(spread)})
    r = pd.DataFrame(rows).dropna(subset=["ret_corr"])
    n_pror = int((r["pror"] < 0.99).sum())

    print(f"成立・株価照合 {len(r)} 件 / 按分発生(按分率<0.99) {n_pror} 件（{n_pror/len(r):.0%}）")
    print(f"按分発生案件の平均按分率 {r.loc[r['pror']<0.99,'pror'].mean():.2f}")

    def stat(x):
        return f"mean {x.mean():+.2%} / median {x.median():+.2%} / SR {x.mean()/x.std():.2f}"

    print("\n--- 全成立案件：フルテンダー vs 按分補正 ---")
    print(f"  フルテンダー前提 : {stat(r['ret_full'])}")
    print(f"  按分補正済み     : {stat(r['ret_corr'])}")
    print(f"  差（補正−前提）  : 平均 {(r['ret_corr']-r['ret_full']).mean():+.2%} / "
          f"最悪 {(r['ret_corr']-r['ret_full']).min():+.2%}")

    print("\n--- スプレッド帯別（按分補正後の平均リターンと按分発生数） ---")
    for b in ["<1%", "1-3%", "3-7%", ">=7%"]:
        g = r[r["bucket"] == b]
        if len(g):
            print(f"  {b:>5}: n={len(g):3d} 按分{int((g['pror']<0.99).sum()):2d}件 "
                  f"フル平均{g['ret_full'].mean():+.2%} 補正平均{g['ret_corr'].mean():+.2%} "
                  f"差{(g['ret_corr']-g['ret_full']).mean():+.2%}")

    print("\n--- 按分が効いた案件（按分率<0.99・補正で最も損なわれた順） ---")
    aff = r[r["pror"] < 0.99].copy()
    aff["hit"] = aff["ret_corr"] - aff["ret_full"]
    print(aff.sort_values("hit")[["sec", "spread", "pror", "ret_full", "ret_corr"]].head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
