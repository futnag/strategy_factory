r"""仮説検証 ③：スマートマネー vs 個人の乖離 → TOPIX タイミング（週次）。

既試の「海外フロー単独→TOPIX」(flow_topix_timing・弱) と違い、**主体間の乖離**を見る：
個人(Ind=逆張り・遅行)が買い越し × 海外/信託(Frgn/TrstBnk=情報を持つ限界買い手)が売り越し
（乖離）の局面は翌週以降 TOPIX が弱い——を判定器で裁く。レジストリ未登録の新規 scope。

シグナル（PubDate基準＝公表日＝PIT、各 intensity=Bal/Tot を過去52週ローリングz）:
- smart    = (z海外 + z信託)/2
- diverg   = smart − z個人   （スマートが個人より強く買う＝強気）
- retail_c = −z個人          （個人が通常より売り＝逆張り強気）
TOPIX(0000) 週次を long/flat でタイミング（side ±1）。K=4。

実行: .venv\Scripts\python.exe examples\research_flow_divergence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import flows  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, SignalTimingStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

OOS = "2024-01"
SECTION = "TokyoNagoya"


def _rz(s: pd.Series, w: int = 52, mp: int = 26) -> pd.Series:
    return (s - s.rolling(w, min_periods=mp).mean()) / s.rolling(w, min_periods=mp).std()


def main() -> int:
    print("=== ③ スマートマネー vs 個人の乖離 → TOPIX タイミング ===")
    inv = flows.load_investor_types()
    sub = inv[inv["Section"] == SECTION].dropna(subset=["PubDate"]).copy()
    sub = sub.set_index("PubDate").sort_index()
    sub = sub[~sub.index.duplicated(keep="last")]

    def intensity(pfx: str) -> pd.Series:
        return (sub[f"{pfx}Bal"] / sub[f"{pfx}Tot"].replace(0, np.nan)).astype(float)

    z_frgn, z_ind = _rz(intensity("Frgn")), _rz(intensity("Ind"))
    z_trst = _rz(intensity("TrstBnk"))
    smart = (z_frgn + z_trst) / 2.0
    diverg = (smart - z_ind).dropna()
    retail_c = (-z_ind).dropna()
    smart_only = smart.dropna()
    print(f"フロー週数 {len(sub)}（{sub.index.min():%Y-%m}〜{sub.index.max():%Y-%m}）")

    # TOPIX 週次
    topix = jq.fetch_index_bars(code="0000").dropna(subset=["Date"])
    tc = topix.set_index("Date")["C"].sort_index()
    tc = tc[~tc.index.duplicated(keep="last")]
    tw = tc.resample("W-FRI").last().dropna()
    view = AsOfView({"close": tw.to_frame("0000")})
    bh = tw.pct_change().dropna()
    print(f"TOPIX週次 {len(tw)} 本  買い持ちSR(ann)={sharpe_ratio(bh) * np.sqrt(52):+.2f}")

    grid = [
        SignalTimingStrategy(diverg, "0000", 0.0, 1, name="flow_diverg(long_smart>retail)"),
        SignalTimingStrategy(diverg, "0000", 0.0, -1, name="flow_diverg(short)"),
        SignalTimingStrategy(retail_c, "0000", 0.0, 1, name="flow_retail_contrarian"),
        SignalTimingStrategy(smart_only, "0000", 0.0, 1, name="flow_smart_only"),
    ]
    with default_registry() as reg:
        v = judge_grid(
            grid, view, scope="flow_divergence_topix",
            hypothesis="個人が買い越し×海外/信託が売り越しの乖離局面で翌週以降TOPIXは弱い（逆も真）。"
                       "単一フローより主体間の乖離が情報量を持つ",
            economic_rationale="個人は逆張り・遅行（dumb money）、海外/信託は情報を持つ限界買い手（smart money）。"
                               "両者の乖離はミスプライスの方向を示す。反対側は需給を読まない参加者。",
            registry=reg, costs_bps=5.0, execution_lag=0)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(52) if is_.size >= 26 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(52) if oos.size >= 26 else np.nan
        print(f"  {r.name:<32} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    print(f"\n  ※ 買い持ちTOPIX SR(ann)={sharpe_ratio(bh) * np.sqrt(52):+.2f} を上回るタイミングのみ価値。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
