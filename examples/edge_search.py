"""実エッジ探索：マイクロ構造特徴で実 BTC/JPY のエッジを探す。

素朴な特徴（bitbank_e2e では DSR=0.62=有意エッジ無し）に対し、OHLCV から計算する
マイクロ構造・流動性特徴（VPIN / Parkinson / Garman-Klass / Amihud / Roll /
Corwin-Schultz / RSI / 多ホライズン・モメンタム）を加え、同じ honest パイプライン
（purged CPCV ＋ 実効サンプル数 ＋ DSR・8設定スイープ）でベースラインと比較する。

honest設計：DSR が改善しなければ「この特徴群でも実エッジ無し」と正直に報告する。
真のアウトオブサンプルは実データでのみ得られる（López de Prado）。

実行（リポジトリ root から）: python examples/edge_search.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from invest_system.backtest.cpcv_backtest import cpcv_backtest  # noqa: E402
from invest_system.data.sources.bitbank import fetch_candlesticks  # noqa: E402
from invest_system.features.frac_diff import find_min_d, frac_diff_ffd  # noqa: E402
from invest_system.features.microstructure import (  # noqa: E402
    amihud_illiquidity,
    corwin_schultz_spread,
    garman_klass_vol,
    parkinson_vol,
    roll_spread,
    rsi,
    vpin,
)
from invest_system.labeling.triple_barrier import (  # noqa: E402
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)
from invest_system.sampling.uniqueness import average_uniqueness  # noqa: E402
from invest_system.validation.cpcv import CombinatorialPurgedKFold  # noqa: E402
from invest_system.validation.registry import TrialRegistry  # noqa: E402

YEARS = ("2021", "2022", "2023", "2024", "2025", "2026")
_CONFIGS = [{"max_depth": d, "max_features": mf, "random_state": s}
            for s, (d, mf) in enumerate(
                [(2, 0.5), (3, 0.5), (3, 1.0), (4, 0.5),
                 (4, 1.0), (5, 0.5), (5, 1.0), (6, 0.5)])]


def evaluate(name, X, y, ret, t1, close):
    """8設定を purged CPCV で評価し、最良の Sharpe 分布と DSR（実効n）を返す。"""
    cv = CombinatorialPurgedKFold(6, 2, embargo_pct=0.01)
    reg = TrialRegistry(":memory:")
    n_eff = int(average_uniqueness(close.index, t1).sum())
    best = {"sr": -np.inf, "uuid": None, "res": None}
    for c in _CONFIGS:
        res = cpcv_backtest(
            X, y, ret, t1, cv,
            lambda c=c: RandomForestClassifier(n_estimators=60, n_jobs=-1, **c))
        tid = reg.preregister(
            scope=name,
            hypothesis=f"{name} RF(depth={c['max_depth']}) predicts 4h direction",
            economic_rationale="microstructure / liquidity / momentum in BTC/JPY",
            params=c)
        reg.record_result(tid, sharpe=res.mean_sharpe, n_obs=n_eff, skew=0.0, kurt=3.0)
        if res.mean_sharpe > best["sr"]:
            best.update(sr=res.mean_sharpe, uuid=tid, res=res)
    return best["res"], reg.deflated_sharpe(best["uuid"]), n_eff


def main() -> None:
    print(f"bitbank 公開APIから 4時間足 BTC/JPY を取得中 ... {YEARS}")
    candles = fetch_candlesticks("btc_jpy", "4hour", YEARS)
    o, h, l, c, v = (candles["open"], candles["high"], candles["low"],
                     candles["close"], candles["volume"])
    dv = c * v
    print(f"取得: {len(c)} 本  {c.index[0].date()} 〜 {c.index[-1].date()}\n")

    min_d, _, _ = find_min_d(c, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2), thresh=1e-4)
    d = min_d if min_d is not None else 0.4

    base = pd.DataFrame({                              # 素朴特徴（ベースライン）
        "fd": frac_diff_ffd(c, d, thresh=1e-4),
        "r1": c.pct_change(1, fill_method=None),
        "r3": c.pct_change(3, fill_method=None),
        "r6": c.pct_change(6, fill_method=None),
        "vol": get_vol(c, 50),
    })
    micro = pd.DataFrame({                             # マイクロ構造・流動性特徴
        "park": parkinson_vol(h, l, 20),
        "gk": garman_klass_vol(o, h, l, c, 20),
        "amihud": amihud_illiquidity(c, dv, 20),
        "roll": roll_spread(c, 20),
        "cs": corwin_schultz_spread(h, l),
        "vpin": vpin(c, v, 50),
        "rsi": rsi(c, 14),
        "r12": c.pct_change(12, fill_method=None),
        "r24": c.pct_change(24, fill_method=None),
    })
    expanded = pd.concat([base, micro], axis=1)
    feats_exp = expanded.dropna()

    # 共通ラベル（拡張特徴の有効区間で揃える）
    t_events = feats_exp.index
    vb = get_vertical_barriers(c, t_events, num_bars=6)
    events = get_events(c, vb.index, [1.5, 1.5], get_vol(c, 50),
                        min_ret=0.0, vertical_barriers=vb)
    bins = get_bins(events, c)
    bins = bins[bins["bin"] != 0]
    idx = bins.index
    y, ret, t1 = bins["bin"].astype(int), bins["ret"], bins["t1"]
    X_base = base.reindex(idx)
    X_exp = feats_exp.reindex(idx)

    print(f"イベント数 {len(idx)}  特徴量: ベースライン {X_base.shape[1]} / 拡張 {X_exp.shape[1]}")
    rb, dsr_b, n_eff = evaluate("baseline", X_base, y, ret, t1, c)
    re_, dsr_e, _ = evaluate("expanded", X_exp, y, ret, t1, c)
    print(f"実効サンプル数(Σ独自性): {n_eff}\n")

    print("=== ベースライン vs マイクロ構造拡張（実BTC/JPY, 8設定の最良）===")
    print(f"{'':14}{'OOS Sharpe平均':>16}{'負パス':>8}{'DSR':>8}")
    print(f"{'素朴特徴':14}{rb.mean_sharpe:>16.4f}{rb.frac_negative:>8.0%}{dsr_b:>8.3f}")
    print(f"{'＋マイクロ構造':14}{re_.mean_sharpe:>16.4f}{re_.frac_negative:>8.0%}{dsr_e:>8.3f}")
    improved = "改善" if dsr_e > dsr_b else "改善せず"
    print(f"\n→ DSR {dsr_b:.3f} → {dsr_e:.3f}（{improved}）")
    if max(dsr_b, dsr_e) >= 0.95:
        print("  有意なエッジの兆候あり → 感度分析・追加検証へ（過学習に警戒）")
    else:
        print("  いずれも DSR<0.95＝有意エッジ無し。これが正直な結果。"
              "\n  次は別ホライズン/別バー(ドルバー)/代替データ/サンプル独自性重みでの再探索。")


if __name__ == "__main__":
    main()
