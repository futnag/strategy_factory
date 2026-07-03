"""仮説検証：トレンド設計の構造比較（単一EMA／バーベル／ラダー／S字 vs 12-1 サイン）。

事前登録＝docs/54（scope=`trend_structure`・K=6・実行前コミット）。3論文の対立主張
（Valeyre: 単一EMAで十分／Etienne et al.: 中間ホライズン冗長＝バーベル／Moskowitz et al.:
トレンド→ポジション写像はS字）を、本番 tsmom_multiasset と同一の 11資産・月末決定→T+1始値・
片道5bps（DP17）で一発比較する。方向は他市場文献で事前固定＝ローカル覗き見なし。

グリッド（docs/54 §2.4 で固定・6セル）:
  tsmom_12m     sign(r_12M)                       ベースライン（統制）
  ema_single    sign(P − EMA_hl78)                H1
  ema_barbell   ½[sign(P−EMA_hl60)+sign(P−EMA_hl500)]  H2
  ema_ladder    mean_hl∈{20,60,125,250,500} sign(P−EMA_hl)  H2統制
  scurve_tanh   tanh(z),  z=r_12M/σ_ann           H3（飽和）
  scurve_bell   z·exp(−z²/4)/0.858                H3（反転）

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_trend_structure.py
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
from invest_system.research.strategies_tsmom import annualized_vol, tsmom_weights, blend_weights  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

KEYS = ["nk225_fut", "sp500", "nasdaq_comp", "gold", "silver", "platinum",
        "wti", "copper", "usdjpy", "eurjpy", "audjpy"]
OOS = "2024-01"
SCOPE = "trend_structure"
VOL_TARGET = 0.10
COSTS_BPS = 5.0
HL_SINGLE = 78            # Valeyre: 最適半減期 ≈78営業日（事前固定）
HL_BARBELL = (60, 500)    # Etienne: 短期＋長期バーベル（中間125を外す）
HL_LADDER = (20, 60, 125, 250, 500)
BELL_NORM = float(np.sqrt(2.0) * np.exp(-0.5))  # z·exp(−z²/4) の最大値 ≈0.858


class _Replay(Strategy):
    """事前計算済み {決定日: ウェイト}（fill価格ビューと併用・本番 tsmom と同機構）。"""

    def __init__(self, weights: dict, name: str, params: dict):
        self._w = weights
        self.name = name
        self.params = params

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def signal_weights(signal_asof: pd.DataFrame, vol_asof: pd.DataFrame,
                   *, vol_target: float = VOL_TARGET) -> dict:
    """{決定日: w = s×(σ_tgt/σ)/N}。tsmom_weights と同一正規化の連続シグナル版。"""
    out: dict[pd.Timestamp, pd.Series] = {}
    for t in signal_asof.index:
        sig = signal_asof.loc[t]
        vol = vol_asof.loc[t] if t in vol_asof.index else pd.Series(dtype="float64")
        w = (sig * (vol_target / vol)).replace([np.inf, -np.inf], np.nan).dropna()
        w = w[w != 0.0]
        if len(w):
            out[t] = w / len(w)
    return out


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
    if r.size < 8 or float(r.std(ddof=1)) == 0.0:  # 建玉ゼロ年＝全月0.0 のガード
        return float("nan")
    return float(sharpe_ratio(r) * np.sqrt(12))


def main() -> int:
    cl = load_external_prices(KEYS, field="close")
    op = load_external_prices(KEYS, field="open")
    if cl.empty or op.empty:
        print("ERROR: data/investers のミラーが見つかりません。")
        return 1
    print(f"=== トレンド設計の構造比較（{cl.index.min():%Y-%m}〜{cl.index.max():%Y-%m}・"
          f"{len(KEYS)}資産・scope={SCOPE}）===")
    _health(cl)

    # --- 意思決定パネル（PIT・本番 tsmom と同一）---
    cl_ff = cl.ffill(limit=7)
    m_close = cl_ff.groupby(cl_ff.index.to_period("M")).tail(1)
    rebal = m_close.index
    vol = annualized_vol(cl, window=63, floor=0.05)
    vol_m = vol.ffill(limit=7).reindex(rebal)

    # --- 約定パネル（DP17: 決定日の翌営業日始値）---
    op_b = op.bfill(limit=3)
    fill_px = op_b.shift(-1).reindex(rebal)
    view = AsOfView({"close": fill_px})

    # --- シグナル（≤t 情報のみ・docs/54 §2.4 固定）---
    def ema_sign(hl: int) -> pd.DataFrame:
        return np.sign(cl_ff - cl_ff.ewm(halflife=hl, min_periods=hl // 2).mean()).reindex(rebal)

    r12 = (m_close / m_close.shift(12) - 1.0)
    z = (r12 / vol_m).replace([np.inf, -np.inf], np.nan)

    signals = {
        "tsmom_12m": np.sign(r12),
        "ema_single": ema_sign(HL_SINGLE),
        "ema_barbell": sum(ema_sign(h) for h in HL_BARBELL) / len(HL_BARBELL),
        "ema_ladder": sum(ema_sign(h) for h in HL_LADDER) / len(HL_LADDER),
        "scurve_tanh": np.tanh(z),
        "scurve_bell": (z * np.exp(-z ** 2 / 4.0)) / BELL_NORM,
    }
    sets = {name: signal_weights(sig, vol_m) for name, sig in signals.items()}
    strategies = [_Replay(w, name, {"design": name, "vol_target": VOL_TARGET,
                                    "universe": len(KEYS), "fills": "t+1_open"})
                  for name, w in sets.items()]

    hyp = ("トレンド設計の構造改善（H1 単一EMA hl78／H2 短長バーベル≥等加重ラダー／"
           "H3 S字写像≥線形サイン）が、同一11資産・同一コストで 12-1 サイン型ベースラインを"
           "ネットで上回る（docs/54・方向は他市場文献で事前固定）")
    rat = ("トレンド情報は単一の支配的タイムスケールに集中し（Valeyre 理論適合 R²≈0.98）、"
           "中間ホライズンは冗長（Etienne）、極端な読みはクラウディングで信頼度低下＝S字"
           "（Moskowitz et al.）。12-1 サインはこの3点を無視した設計＝改善余地が理論的に特定済み")

    with default_registry() as reg:
        v = judge_grid(strategies, view, scope=SCOPE, hypothesis=hyp,
                       economic_rationale=rat, registry=reg, costs_bps=COSTS_BPS)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{SCOPE}.html"))

    # ---- 診断（throwaway・K不変）----
    base = v.series.get("tsmom_12m", pd.Series(dtype="float64")).dropna()
    print(f"\n--- 診断: IS/OOS({OOS}〜)・前後2020・回転・コスト感応（5→10/15bps 解析換算）---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        turn = backtest(next(st for st in strategies if st.name == r.name),
                        view, costs_bps=COSTS_BPS).turnover
        t_al = turn.reindex(s.index).fillna(0.0)
        net10 = s - 5.0 / 1e4 * t_al
        net15 = s - 10.0 / 1e4 * t_al
        print(f"  {r.name:<12} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f} | 回転={turn.mean():.2f} | "
              f"@10bps={_sr(net10):+.2f} @15bps={_sr(net15):+.2f}")

    # H1/H2/H3 の対比較（事前登録 §2.6-3）
    print("\n--- 仮説別の対比較（ネット年率SR差・全期間）---")
    pairs = [("H1 単一EMA vs 12-1", "ema_single", "tsmom_12m"),
             ("H2 バーベル vs ラダー", "ema_barbell", "ema_ladder"),
             ("H3 tanh vs 12-1", "scurve_tanh", "tsmom_12m"),
             ("H3 bell vs 12-1", "scurve_bell", "tsmom_12m")]
    for label, a, b in pairs:
        sa, sb = v.series.get(a), v.series.get(b)
        if sa is None or sb is None:
            continue
        d = (sa - sb).dropna()
        npos = int((d > 0).sum())
        print(f"  {label:<22} ΔSR={_sr(sa) - _sr(sb):+.2f}  "
              f"月次勝率={npos}/{len(d)}（差分系列SR={_sr(d):+.2f}）")

    # 本番 tsmom_blend（3/6/12等加重）との相関＝採用提案時の代替/併載判断材料
    prod_sets = [tsmom_weights(m_close, vol_m, lb, vol_target=VOL_TARGET) for lb in (3, 6, 12)]
    prod = _Replay(blend_weights(prod_sets), "prod_blend", {})
    prod_ret = backtest(prod, view, costs_bps=COSTS_BPS).returns.dropna()
    best = v.best.name if v.best else "tsmom_12m"
    sbest = v.series.get(best, pd.Series(dtype="float64")).dropna()
    print(f"\n--- 本番 blend(3/6/12) との相関 ---")
    print(f"  本番blend SR={_sr(prod_ret):+.2f} / 最良セル {best} SR={_sr(sbest):+.2f} / "
          f"月次相関={sbest.corr(prod_ret):+.2f}")

    # 年次SR（最良セル）と WTI 2020-03/04 寄与
    print(f"\n--- 年次 net SR（{best}）---")
    for y, seg in sbest.groupby(sbest.index.year):
        print(f"  {y}: SR={_sr(seg):+.2f}  n={seg.size}")
    fwd = fill_px.pct_change().shift(-1)
    apr20 = 0.0
    for t, w in sets[best].items():
        if t.strftime("%Y-%m") in ("2020-03", "2020-04") and "wti" in w.index:
            x = w["wti"] * fwd.loc[t].get("wti", np.nan)
            if pd.notna(x):
                apr20 += float(x)
    print(f"  WTI 2020-03/04 寄与（{best}）: {apr20:+.2%}")

    print("\n※ 判定は scope=trend_structure の DSR（K=6・docs/54 §2.6）。診断は throwaway（K不変）。"
          "\n※ ベースライン超えでも本番置換は自動でない（採用提案＝人間ゲート）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
