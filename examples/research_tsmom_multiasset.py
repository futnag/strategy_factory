"""仮説検証：マルチアセット時系列モメンタム（TSMOM・柱E＝第4の独立アルファ源候補）。

経済的根拠（Chan の問い）：マクロの大変化（金利サイクル・インフレ転換）は数ヶ月かけて
価格に織り込まれる（情報の漸進的拡散）。反対側には「価格に反応して行動する主体」＝
含み損の投げ・リスク管理の強制縮小・中央銀行の介入が恒常的に存在する。クロスセクション・
モメンタム（銘柄間比較・日本株で棄却済み §6.7）とは別現象で、Moskowitz–Ooi–Pedersen
(2012) が58資産×100年で文書化した最も頑健級のアノマリー。**狙いは単体の高 Sharpe では
なく、value↔PEAD switch（日本株 LS・平時に勝ち危機に弱い）と逆の非対称性を持つ低相関
スリーブ＝breadth（IR≈IC×√breadth・§6.5）**。

ユニバース（事前登録・11資産）：nk225_fut, sp500, nasdaq_comp, gold, silver, platinum,
wti, copper, usdjpy, eurjpy, audjpy。**natgas はデータ品質で除外**（未調整の限月継続足
にロール痕＝対になった±15-47%の跳ねが多数で投資家リターンを測れない。健全性チェックで
建玉前に判定）。重複系列（TOPIX/ダウ/US Cash 等＝同一ベットの二重計上）も a priori 除外。

執行（DP17 準拠＝判定数値そのものが T+1 始値約定）：意思決定は月末終値（PIT）、約定は
**翌営業日の始値**。実装は「fill 価格ビュー」：ビューの close パネルに各決定日の翌営業日
始値（祝日は3日以内の翌始値）を入れ、Replay 戦略（ビューを読まない＝事前計算ウェイト）を
engine lag=0 で回す＝実現は fill(t)→fill(t+1) の open→open。コストは片道5bps
（先物実勢1-2bp＋ロール償却の保守値）、10bps 感応度も併記。

既知の限界（正直に）：investing.com の限月継続足は back-adjust されておらず、ロール時の
跳ねがリターンに混入する（特に WTI 2020-04 の期近崩壊）。サイン・シグナルは頑健だが
P&L はキャリー分だけ歪む。診断で WTI 2020-04 の寄与を分離表示する。

実行: .venv\\Scripts\\python.exe examples\\research_tsmom_multiasset.py
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

from invest_system.data.external import load_external_prices  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, Strategy, backtest, judge_grid, write_html,
)
from invest_system.research.strategies_tsmom import (  # noqa: E402
    annualized_vol, blend_weights, tsmom_weights,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

KEYS = ["nk225_fut", "sp500", "nasdaq_comp", "gold", "silver", "platinum",
        "wti", "copper", "usdjpy", "eurjpy", "audjpy"]
OOS = "2024-01"
SCOPE = "tsmom_multiasset"
LOOKBACKS = (3, 6, 12)
VOL_TARGET = 0.10
COSTS_BPS = 5.0


class _Replay(Strategy):
    """事前計算済み {決定日: ウェイト} を返す（ビューは読まない＝fill価格ビューと併用可）。"""

    def __init__(self, weights: dict, name: str, params: dict):
        self._w = weights
        self.name = name
        self.params = params

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _health(cl: pd.DataFrame) -> None:
    print(f"{'key':<12} {'rows':>5} {'start':>11} {'end':>11} {'|r|>8%':>6}")
    for k in cl.columns:
        s = cl[k].dropna()
        r = s.pct_change().dropna()
        print(f"{k:<12} {len(s):>5} {str(s.index.min().date()):>11} "
              f"{str(s.index.max().date()):>11} {int((r.abs() > .08).sum()):>6}")


def _sr(x: pd.Series, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


def main() -> int:
    cl = load_external_prices(KEYS, field="close")
    op = load_external_prices(KEYS, field="open")
    if cl.empty or op.empty:
        print("ERROR: data/investers のミラーが見つかりません。")
        return 1
    print(f"=== マルチアセット TSMOM（{cl.index.min():%Y-%m}〜{cl.index.max():%Y-%m}・"
          f"{len(KEYS)}資産・scope={SCOPE}）===")
    print("健全性（natgas はロール痕により事前除外済み・docstring 参照）:")
    _health(cl)

    # --- 意思決定パネル（PIT）：月末の最終ユニオン営業日・各資産は直近既知の終値 ---
    cl_ff = cl.ffill(limit=7)
    m_close = cl_ff.groupby(cl_ff.index.to_period("M")).tail(1)
    rebal = m_close.index
    vol = annualized_vol(cl, window=63, floor=0.05)
    vol_m = vol.ffill(limit=7).reindex(rebal)

    # --- 約定パネル（DP17）：決定日の翌営業日始値（祝日は3日以内の翌始値へ繰延）---
    op_b = op.bfill(limit=3)
    fill_px = op_b.shift(-1).reindex(rebal)      # 各決定日 t → t+1営業日の始値
    view = AsOfView({"close": fill_px})

    # --- 戦略（事前登録の小格子：L∈{3,6,12} ＋ 等加重ブレンド ＝ K=4）---
    sets = {f"tsmom_{lb}m": tsmom_weights(m_close, vol_m, lb, vol_target=VOL_TARGET)
            for lb in LOOKBACKS}
    sets["tsmom_blend"] = blend_weights(list(sets.values()))
    strategies = [_Replay(w, name, {"lookback": name, "vol_target": VOL_TARGET,
                                    "universe": len(KEYS), "fills": "t+1_open"})
                  for name, w in sets.items()]

    with default_registry() as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("マルチアセット時系列モメンタム（サイン×ボラターゲット）は、"
                        "情報の漸進的拡散により正の期待値を持ち、日本株 value 系と低相関の"
                        "独立スリーブとして breadth を押し上げるか"),
            economic_rationale=("マクロ変化は数ヶ月かけて織り込まれ、反対側には損切り・強制"
                                "デレバ・介入という価格非感応の主体が恒常的にいる（MOP2012・"
                                "100年級の頑健性）。狙いは単体 Sharpe でなく危機時に勝つ"
                                "非対称性＝旗艦との分散。natgas はロール痕で事前除外（PIT）。"),
            registry=reg, costs_bps=COSTS_BPS)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # --- IS/OOS・レバレッジ・コスト感応度 ---
    fwd = fill_px.pct_change().shift(-1)         # 決定日 t の実現（fill→fill）
    by_name = {st.name: st for st in strategies}
    print(f"\n--- IS/OOS（保留 {OOS}〜）・グロスレバ・10bps感応 ---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        gross_lev = pd.Series({t: float(w.abs().sum())
                               for t, w in sets[r.name].items()})
        # 10bps の解析的再計算：net10 = net5 − 5bp×回転率
        turn = backtest(by_name[r.name], view, costs_bps=COSTS_BPS).turnover
        net10 = s - 5.0 / 1e4 * turn.reindex(s.index).fillna(0.0)
        print(f"  {r.name:<12} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f} | レバ平均{gross_lev.mean():.2f}/"
              f"最大{gross_lev.max():.2f} | SR@10bps={_sr(net10):+.2f}")

    # --- 分散価値の診断：株式市場との相関・ストレス窓・資産別寄与 ---
    blend = v.series.get("tsmom_blend", pd.Series(dtype="float64")).dropna()
    mret = fwd[["nk225_fut", "sp500"]].reindex(blend.index)
    print("\n--- 旗艦との相性診断（tsmom_blend）---")
    print(f"  月次相関: vs 日経先物 {blend.corr(mret['nk225_fut']):+.2f} / "
          f"vs S&P500 {blend.corr(mret['sp500']):+.2f}")
    for label, lo, hi in [("2018Q4(米株急落)", "2018-10", "2019-01"),
                          ("2020-02..03(COVID)", "2020-02", "2020-04"),
                          ("2022(金利急騰)", "2022-01", "2023-01")]:
        seg = blend[(blend.index >= lo) & (blend.index < hi)]
        nk = mret["nk225_fut"][(mret.index >= lo) & (mret.index < hi)]
        print(f"  {label:<20} TSMOM {float((1 + seg).prod() - 1):+7.2%} / "
              f"日経 {float((1 + nk).prod() - 1):+7.2%}")

    contrib = {}
    for t, w in sets["tsmom_blend"].items():
        f = fwd.loc[t].reindex(w.index)
        for k_, x in (w * f).dropna().items():
            contrib.setdefault(k_, []).append(x)
    print("\n--- 資産別寄与（tsmom_blend・年率平均リターン寄与）---")
    rows = sorted(((k_, np.mean(xs) * 12, len(xs)) for k_, xs in contrib.items()),
                  key=lambda z: -z[1])
    for k_, c, n in rows:
        print(f"  {k_:<12} {c:+7.2%}（建玉月 {n}）")
    apr20 = 0.0
    for t, w in sets["tsmom_blend"].items():
        if t.strftime("%Y-%m") in ("2020-03", "2020-04") and "wti" in w.index:
            x = w["wti"] * fwd.loc[t].get("wti", np.nan)
            if pd.notna(x):
                apr20 += float(x)
    print(f"  ※ WTI 2020-03/04（期近崩壊・継続足の歪み源）の寄与: {apr20:+.2%}"
          f"（除外しても結論が変わらないか目視確認）")

    print("\n※ 判定は scope 累計 K のデフレートDSR。狙いは単体認定でなく低相関スリーブ＝"
          "旗艦（value↔PEAD switch）との合成改善は両系列が揃う次段で別 scope として裁く。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
