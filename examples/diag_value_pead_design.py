"""value+PEAD の設計診断：保有期間・分散・リスク管理（使い捨て・K不変・判定なし）。

docs/26-28・41-43 の帰結＝「レジームは当てられない／因果調整でも非定常は消えない＝予測より
防御」を受け、**タイミングに賭けず、設計で非定常を吸収する**3レバーを実証する：

  レバー1 保有期間：建玉を H か月オーバーラップ保有（回転 ~1/H）→ コスト/安定への効果。
  レバー2 分散    ：(a) 因子構成（value単独/PEAD単独/value+PEAD/＋quality＋mom）、
                    (b) 銘柄分散（分位 0.1/0.2/0.3）→ ボラ低減・worst-epoch 改善。
  レバー3 リスク管理：(a) ボラ・ターゲティング（自己トレーリングボラで露出調整・PIT）、
                    (b) 暴落ディフェンス（市場高ボラ局面で露出半減＝docs/28 の唯一の用途）。

phase4 月次横断面（book_to_market/earnings_yield=value, sue/forecast=PEAD, roe/
gross_prof/accruals=quality, momentum）を基盤。すべて data/ ローカルのみ・PIT・先読み
なし。**これは設計トレードオフの記述であり、最良構成を選ぶ DSR 判定ではない**
（最終構成は別途 FROZEN 事前登録→一度きり判定が必要）。worst-epoch 頑健性を重視。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_causal_adjustment import EPOCHS, cs_standardize  # noqa: E402

DATA = ROOT / "data" / "phase4"
COST_BPS = 20.0   # 片道 bps（回転に課金）
ANN = np.sqrt(12)


# ----------------------------------------------------------------- 指標
def sharpe(r): return r.mean() / r.std() * ANN if r.std() > 0 else np.nan
def ann_ret(r): return r.mean() * 12
def ann_vol(r): return r.std() * ANN
def maxdd(r):
    c = (1 + r.fillna(0)).cumprod()
    return float((c / c.cummax() - 1).min())


def epoch_sharpes(r):
    out = []
    for _, a, b in EPOCHS:
        sub = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
        out.append(sharpe(sub) if len(sub) >= 6 else np.nan)
    return out


# ----------------------------------------------------- LS ポートフォリオ
def ls_weights(sig_wide: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """各月、上位/下位 quantile を等加重 L/S（long合計+1, short合計-1）。"""
    W = pd.DataFrame(0.0, index=sig_wide.index, columns=sig_wide.columns)
    for dt, row in sig_wide.iterrows():
        fr = row.dropna()
        if len(fr) < 20:
            continue
        k = max(1, int(len(fr) * quantile))
        order = fr.sort_values()
        W.loc[dt, order.index[-k:]] = 1.0 / k
        W.loc[dt, order.index[:k]] = -1.0 / k
    return W


def portfolio(sig_wide, ret_wide, quantile=0.2, hold=1, costs_bps=COST_BPS):
    """オーバーラップ H か月保有の LS ネットリターン・回転を返す。
    w_held[t] = 直近 H か月の目標 w の平均（回転 ~1/H）。"""
    W = ls_weights(sig_wide, quantile)
    if hold > 1:
        W = W.rolling(hold, min_periods=1).mean()
    common = W.index.intersection(ret_wide.index)
    W, R = W.loc[common], ret_wide.loc[common]
    gross = (W * R.reindex(columns=W.columns)).sum(axis=1)
    dW = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    cost = (costs_bps / 1e4) * dW
    net = (gross - cost).replace(0.0, np.nan).dropna()
    return net, dW.reindex(net.index)


def _fmt_ep(es):
    return "".join(f"{x:>+7.2f}" if np.isfinite(x) else f"{'—':>7}" for x in es)


# ===========================================================================
def main() -> int:
    warnings.filterwarnings("ignore")
    print("=" * 84)
    print("value+PEAD 設計診断：保有期間・分散・リスク管理（K不変・判定なし・PIT）")
    print("=" * 84)
    print(f"コスト片道 {COST_BPS:.0f}bps / エポック: "
          + " ".join(e[0][:6] for e in EPOCHS))

    X = pd.read_parquet(DATA / "X.parquet")
    y = pd.read_parquet(DATA / "y.parquet")["y"]
    Xz = cs_standardize(X)

    # --- 因子シグナル（rank-z・高い=ロング）-------------------------------
    def z(c): return Xz[c]
    value = pd.concat([z("book_to_market"), z("earnings_yield")], axis=1).mean(axis=1)
    pead = pd.concat([z("sue_recent"), z("sue_initial"),
                      z("forecast_revision")], axis=1).mean(axis=1)
    quality = pd.concat([z("roe"), z("gross_profitability"),
                         -z("accruals")], axis=1).mean(axis=1)
    mom = z("momentum")
    sigs = {
        "value単独": value,
        "PEAD単独": pead,
        "value+PEAD": pd.concat([value, pead], axis=1).mean(axis=1),
        "+quality+mom": pd.concat([value, pead, quality, mom], axis=1).mean(axis=1),
    }
    # wide 化（month × Code）
    def to_wide(s): return s.unstack("Code")
    ret_wide = to_wide(y)
    sig_wide = {k: to_wide(v) for k, v in sigs.items()}

    # ベース＝value+PEAD・分位0.2・H=1
    base = "value+PEAD"

    # =====================================================================
    # レバー2(a) 因子分散：構成の比較（H=1, q=0.2）
    # =====================================================================
    print("\n" + "=" * 84)
    print("レバー2(a) 因子分散：シグナル構成の比較（H=1・分位0.2・ネット）")
    print("=" * 84)
    print(f"{'構成':14}{'SR':>6}{'年率':>7}{'ボラ':>6}{'maxDD':>7}{'回転':>6}"
          f"   エポック別SR(A/B/C/D)")
    for k in sigs:
        net, tov = portfolio(sig_wide[k], ret_wide)
        es = epoch_sharpes(net)
        print(f"{k:14}{sharpe(net):>6.2f}{ann_ret(net):>+7.1%}{ann_vol(net):>6.1%}"
              f"{maxdd(net):>7.1%}{tov.mean()*12:>6.1f}   {_fmt_ep(es)}  "
              f"worst={np.nanmin(es):+.2f}")
    print("  → 分散（多因子）が worst-epoch SR を引き上げるか／全期間SRと頑健性の対比。")

    # =====================================================================
    # レバー1 保有期間：H か月オーバーラップ（value+PEAD・q=0.2）
    # =====================================================================
    print("\n" + "=" * 84)
    print(f"レバー1 保有期間：H か月保有（{base}・分位0.2・回転~1/H）")
    print("=" * 84)
    print(f"{'H(月)':>6}{'SR(gross)':>10}{'SR(net)':>9}{'年率net':>8}{'maxDD':>7}"
          f"{'回転/年':>8}   エポック別SR(net)")
    for H in [1, 3, 6, 12]:
        net, tov = portfolio(sig_wide[base], ret_wide, hold=H)
        gnet, _ = portfolio(sig_wide[base], ret_wide, hold=H, costs_bps=0.0)
        es = epoch_sharpes(net)
        print(f"{H:>6}{sharpe(gnet):>10.2f}{sharpe(net):>9.2f}{ann_ret(net):>+8.1%}"
              f"{maxdd(net):>7.1%}{tov.mean()*12:>8.1f}   {_fmt_ep(es)}")
    print("  → H↑で回転↓＝コスト drag↓。ネットSRが改善する H と、過度の希薄化の境目。")

    # =====================================================================
    # レバー2(b) 銘柄分散：分位の比較（value+PEAD・H=1）
    # =====================================================================
    print("\n" + "=" * 84)
    print(f"レバー2(b) 銘柄分散：分位（{base}・H=1）— 広いほど分散・浅いシグナル")
    print("=" * 84)
    print(f"{'分位':>6}{'SR(net)':>9}{'年率':>7}{'ボラ':>6}{'maxDD':>7}{'回転/年':>8}"
          f"   エポック別SR(net)")
    for q in [0.1, 0.2, 0.3]:
        net, tov = portfolio(sig_wide[base], ret_wide, quantile=q)
        es = epoch_sharpes(net)
        print(f"{q:>6.1f}{sharpe(net):>9.2f}{ann_ret(net):>+7.1%}{ann_vol(net):>6.1%}"
              f"{maxdd(net):>7.1%}{tov.mean()*12:>8.1f}   {_fmt_ep(es)}")
    print("  → 分位を広げると分散↑・ボラ↓だがシグナル希薄化。worst-epoch とのバランス。")

    # =====================================================================
    # レバー3 リスク管理：ボラ・ターゲティング & 暴落ディフェンス
    # =====================================================================
    print("\n" + "=" * 84)
    print(f"レバー3 リスク管理（{base}・H=3・分位0.2 をベースに）")
    print("=" * 84)
    net, tov = portfolio(sig_wide[base], ret_wide, hold=3)

    # (a) ボラ・ターゲティング：自己トレーリング12mボラ→目標年率10%（PIT・shift）
    tgt = 0.10
    rvol = net.rolling(12, min_periods=6).std().shift(1) * ANN
    scale = (tgt / rvol).clip(upper=3.0)
    net_vt = (net * scale).dropna()

    # (b) 暴落ディフェンス：市場(横断面平均)の拡張窓ボラ三分位が最上位の翌月、露出半減
    mkt = ret_wide.mean(axis=1).reindex(net.index)
    mvol = mkt.rolling(6, min_periods=3).std()
    from invest_system.timeseries.regime import expanding_tertile
    hi = (expanding_tertile(mvol, min_periods=24) == 2).shift(1).fillna(False)
    net_def = net.copy()
    net_def[hi.reindex(net_def.index).fillna(False)] *= 0.5

    print(f"{'手法':18}{'SR':>6}{'年率':>7}{'ボラ':>6}{'maxDD':>7}"
          f"   エポック別SR")
    for nm, r in [("ベース(H=3)", net), ("+ボラ目標10%", net_vt),
                  ("+暴落ディフェンス", net_def)]:
        es = epoch_sharpes(r)
        print(f"{nm:18}{sharpe(r):>6.2f}{ann_ret(r):>+7.1%}{ann_vol(r):>6.1%}"
              f"{maxdd(r):>7.1%}   {_fmt_ep(es)}  worst={np.nanmin(es):+.2f}")
    print("  → ボラ目標は SR をならし maxDD を圧縮。暴落ディフェンスは B(COVID) の DD 緩和。")

    print("\n" + "=" * 84)
    print("注：本診断は設計トレードオフの記述（K不変・先読みなし）。最良構成の選択では")
    print("ない。実運用化は別途 FROZEN 事前登録→一度きり DSR 判定（docs/25 系）が必要。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
