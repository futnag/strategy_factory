r"""仮説検証 ①：オプション・インプライドのリスク・レジーム → TOPIX タイミング。

日経225オプション（options_225）から日次の implied リスク指標を構築し、TOPIX(0000)を
タイミング建玉できるかを検証ファクトリで裁く。**ボラ"売り"(VRP, vol_premium_n225)とは別物**＝
ここは方向・タイミング/ゲートとしての利用。レジストリ未登録の新規 scope。

日次 implied 指標（各営業日・前場引け基準）:
- atm_iv : 直近限月の ATM（moneyness≈1）IV 中央値（水準）
- skew   : 直近限月のダウンサイド（moneyness 0.87〜0.93）IV − ATM IV（プット・リッチ＝恐怖）
- term   : 直近限月 ATM IV − 次限月 ATM IV（>0＝バックワーデーション＝急性ストレス）
これらを過去252日でローリング標準化（PIT・先読みなし）し、合成リスクと skew を作る。

2方向を検証（SignalTimingStrategy は long/flat）:
- *_calm  : リスクが過去平均より低い（calm）局面でロング（リスクオフ回避）
- *_stress: リスクが過去平均より高い（stress）局面でロング（恐怖の買い／VRP・平均回帰）

実行: .venv\Scripts\python.exe examples\research_option_regime.py
"""
from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, SignalTimingStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
OPT_DIR = Path("data/jquants/options_225")
SIG_CACHE = Path("data/processed/option_n225_signals.parquet")
COSTS_BPS = 5.0          # 指数(ETF/先物)タイミングの低コスト目安


def _day_sig(d: pd.DataFrame) -> dict | None:
    """1営業日の全契約 → {atm_iv, skew, term}（put/call は moneyness 帯で回避）。"""
    u = d["UnderPx"].dropna()
    if u.empty:
        return None
    underpx = float(u.iloc[0])
    d = d[(d["IV"] > 3) & (d["IV"] < 150) & (d["OI"] > 0)].copy()
    if len(d) < 10 or underpx <= 0:
        return None
    d["mny"] = d["Strike"] / underpx
    cms = d.dropna(subset=["SQD"]).groupby("CM")["SQD"].min().sort_values()
    if cms.empty:
        return None

    def atmiv(cm: str) -> float:
        s = d[d["CM"] == cm]
        near = s[(s["mny"] - 1.0).abs() <= 0.03]
        if near.empty and len(s):
            near = s.iloc[(s["mny"] - 1.0).abs().to_numpy().argsort()[:6]]
        return float(near["IV"].median()) if len(near) else float("nan")

    front = cms.index[0]
    fa = atmiv(front)
    na = atmiv(cms.index[1]) if len(cms) >= 2 else float("nan")
    fr = d[d["CM"] == front]
    dn = fr[(fr["mny"] >= 0.87) & (fr["mny"] <= 0.93)]
    skew = (float(dn["IV"].median()) - fa) if len(dn) >= 2 and not np.isnan(fa) else float("nan")
    term = fa - na
    if np.isnan(fa):
        return None
    return {"atm_iv": fa, "skew": skew, "term": term}


def build_option_signals(refresh: bool = False) -> pd.DataFrame:
    """日次 implied 指標を構築（2,600+ 日次ファイルを走査）。結果は parquet にキャッシュ。"""
    if SIG_CACHE.exists() and not refresh:
        return pd.read_parquet(SIG_CACHE)
    d0, d1 = pd.Timestamp(START), pd.Timestamp(END) + pd.offsets.MonthEnd(0)
    cols = ["CM", "Strike", "IV", "UnderPx", "OI", "SQD"]
    rows = []
    files = sorted(glob(str(OPT_DIR / "*.parquet")))
    for fp in files:
        stem = Path(fp).stem
        try:
            ts = pd.to_datetime(stem, format="%Y%m%d")
        except ValueError:
            continue
        if not (d0 <= ts <= d1):
            continue
        try:
            df = pd.read_parquet(fp, columns=cols)
        except Exception:           # _empty マーカー（祝日・データ無し）→ 読み飛ばす
            continue
        sig = _day_sig(df)
        if sig:
            rows.append({"Date": ts, **sig})
    out = pd.DataFrame(rows).set_index("Date").sort_index()
    SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SIG_CACHE)
    return out


def _rz(x: pd.Series, w: int = 252, mp: int = 120) -> pd.Series:
    """過去 w 日のローリング標準化（PIT・先読みなし）。"""
    mu = x.rolling(w, min_periods=mp).mean()
    sd = x.rolling(w, min_periods=mp).std()
    return (x - mu) / sd.replace(0.0, np.nan)


def main() -> int:
    print(f"=== ① オプション・レジーム → TOPIX タイミング {START}〜{END} ===")
    sig = build_option_signals()
    print(f"implied 指標 {len(sig)} 営業日（{sig.index.min():%Y-%m}〜{sig.index.max():%Y-%m}）"
          f"  キャッシュ: {SIG_CACHE}")

    # 合成リスク（3指標の平均ローリングz）と skew 単体
    risk_comp = pd.concat([_rz(sig["skew"]), _rz(sig["term"]), _rz(sig["atm_iv"])],
                          axis=1).mean(axis=1)
    risk_skew = _rz(sig["skew"])

    # TOPIX 日次
    topix = jq.fetch_index_bars(code="0000").dropna(subset=["Date"])
    tc = topix.set_index("Date")["C"].sort_index()
    tc = tc[~tc.index.duplicated(keep="last")]
    tdf = tc.to_frame("0000")
    tdf.index = pd.to_datetime(tdf.index).normalize()
    # ウォームアップ後から評価（先頭の現金プレフィックスを避ける）
    first = risk_comp.dropna().index.min()
    tdf = tdf.loc[first:]
    view = AsOfView({"close": tdf})
    bh = tc.pct_change().dropna()
    bh = bh[bh.index >= first]
    print(f"TOPIX 日次 {len(tdf)} 本（{tdf.index.min():%Y-%m}〜{tdf.index.max():%Y-%m}）  "
          f"買い持ちSR(ann)={sharpe_ratio(bh) * np.sqrt(252):+.2f}")

    # シグナル（前日基準＝.shift(1) で当日に利用可能化＝PIT）
    measures = {"comp": risk_comp, "skew": risk_skew}
    grid = []
    for nm, risk in measures.items():
        calm = (-risk).shift(1).dropna()
        stress = risk.shift(1).dropna()
        grid.append(SignalTimingStrategy(calm, "0000", threshold=0.0, side=1,
                                         name=f"optregime_{nm}_calm"))
        grid.append(SignalTimingStrategy(stress, "0000", threshold=0.0, side=1,
                                         name=f"optregime_{nm}_stress"))
    # 各シグナルの建玉率（exposure）= 解釈用
    for nm, risk in measures.items():
        inv_calm = float(((-risk).shift(1) > 0).mean())
        print(f"  {nm}: calm建玉率≈{inv_calm:.0%} / stress建玉率≈{1 - inv_calm:.0%}")

    with default_registry() as reg:
        v = judge_grid(
            grid, view, scope="option_regime_topix",
            hypothesis="日経225オプションのimplied リスク（スキュー＋ターム構造＋IV水準）は将来の"
                       "TOPIXの方向/リスクを先読みし、タイミング建玉で買い持ちを上回る",
            economic_rationale="オプション市場はテールリスクを価格化し、プット・スキューと"
                               "ターム構造のバックワーデーションは確立した先行ストレス指標。"
                               "実現ボラ・ゲート（既試）と違いフォワードルッキング。需給/価格と別系統の時系列信号。",
            registry=reg, costs_bps=COSTS_BPS, execution_lag=0)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）・買い持ち比較 ---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(252) if is_.size >= 60 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(252) if oos.size >= 60 else np.nan
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    print(f"\n  ※ 買い持ちTOPIX SR(ann)={sharpe_ratio(bh) * np.sqrt(252):+.2f} を上回るタイミングのみ価値。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
