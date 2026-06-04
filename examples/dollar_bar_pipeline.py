"""ドルバー上での honest パイプライン（実エッジ探索A・後半）。

実tickから真のドルバーを構築し、その上で特徴量→トリプルバリア→purged CPCV→DSR を
回す。OHLCV足ベースライン（edge_search: DSR≈0.30）と比較し、統計特性の良い
ドルバー＋tick固有の実オーダーフロー特徴がエッジ評価を変えるかを検証する。

tick固有の価値：各ドルバーの「実際の売買符号による純オーダーフロー不均衡」を特徴量化
（OHLCV では得られない）。

実行（リポジトリ root から）: python examples/dollar_bar_pipeline.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from invest_system.backtest.cpcv_backtest import cpcv_backtest  # noqa: E402
from invest_system.data.bars import dollar_bars  # noqa: E402
from invest_system.data.sources.bitbank import fetch_transactions  # noqa: E402
from invest_system.features.frac_diff import find_min_d, frac_diff_ffd  # noqa: E402
from invest_system.features.microstructure import (  # noqa: E402
    amihud_illiquidity,
    garman_klass_vol,
    parkinson_vol,
    rsi,
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

N_DAYS = 60
_CONFIGS = [{"max_depth": d, "max_features": mf, "random_state": s}
            for s, (d, mf) in enumerate(
                [(2, 0.5), (3, 0.5), (3, 1.0), (4, 0.5),
                 (4, 1.0), (5, 0.5), (5, 1.0), (6, 0.5)])]


def order_flow_per_bar(tx: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """各ドルバーの純オーダーフロー不均衡比 ∈[-1,1]（実約定 side を使用）。"""
    bar_t = bars.index.asi8
    tick_t = tx.index.asi8
    idx = np.searchsorted(bar_t, tick_t, side="left")        # tick→所属バー
    sign = np.where(tx["side"].to_numpy() == "buy", 1.0, -1.0)
    vol = tx["volume"].to_numpy(dtype=float)
    valid = idx < len(bars)
    net = pd.Series((vol * sign)[valid]).groupby(idx[valid]).sum()
    tot = pd.Series(vol[valid]).groupby(idx[valid]).sum()
    flow = (net / tot).reindex(range(len(bars))).fillna(0.0)
    flow.index = bars.index
    return flow


def evaluate(X, y, ret, t1, close):
    cv = CombinatorialPurgedKFold(6, 2, embargo_pct=0.01)
    reg = TrialRegistry(":memory:")
    n_eff = int(average_uniqueness(close.index, t1).sum())
    best = {"sr": -np.inf, "uuid": None, "res": None}
    for c in _CONFIGS:
        res = cpcv_backtest(
            X, y, ret, t1, cv,
            lambda c=c: RandomForestClassifier(n_estimators=60, n_jobs=-1, **c))
        tid = reg.preregister(scope="dollarbar",
                              hypothesis=f"RF(depth={c['max_depth']}) on dollar bars",
                              economic_rationale="order-flow / microstructure on dollar bars")
        reg.record_result(tid, sharpe=res.mean_sharpe, n_obs=n_eff, skew=0.0, kurt=3.0)
        if res.mean_sharpe > best["sr"]:
            best.update(sr=res.mean_sharpe, uuid=tid, res=res)
    return best["res"], reg.deflated_sharpe(best["uuid"]), n_eff


def main() -> None:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    dates = [(end - timedelta(days=i)).strftime("%Y%m%d") for i in range(N_DAYS - 1, -1, -1)]
    print(f"bitbank から約定(tick)を取得中 ... {dates[0]}〜{dates[-1]}（{N_DAYS}日）")
    tx = fetch_transactions("btc_jpy", dates, pause=0.2)
    dv_total = float((tx["price"] * tx["volume"]).sum())
    threshold = dv_total / (N_DAYS * 30)                     # 約30バー/日
    dbar = dollar_bars(tx, threshold)
    dbar = dbar[~dbar.index.duplicated(keep="last")]   # 同一msで複数バーが閉じる稀な重複を除去
    close = dbar["close"]
    print(f"取得: {len(tx):,} 約定 → ドルバー {len(dbar)} 本 "
          f"({dbar.index[0]} 〜 {dbar.index[-1]})\n")

    flow = order_flow_per_bar(tx, dbar)
    d = find_min_d(close, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2), thresh=1e-3)[0] or 0.4
    feats = pd.DataFrame({
        "fd": frac_diff_ffd(close, d, thresh=1e-3),
        "r1": close.pct_change(1, fill_method=None),
        "r3": close.pct_change(3, fill_method=None),
        "r6": close.pct_change(6, fill_method=None),
        "vol": get_vol(close, 50),
        "park": parkinson_vol(dbar["high"], dbar["low"], 20),
        "gk": garman_klass_vol(dbar["open"], dbar["high"], dbar["low"], close, 20),
        "amihud": amihud_illiquidity(close, dbar["dollar"], 20),
        "rsi": rsi(close, 14),
        "flow": flow,                                        # tick固有：実オーダーフロー
        "vpin": flow.abs().rolling(50).mean(),               # tick固有：実VPIN
    }).dropna()

    t_events = feats.index
    vb = get_vertical_barriers(close, t_events, num_bars=10)
    events = get_events(close, vb.index, [1.5, 1.5], get_vol(close, 50),
                        min_ret=0.0, vertical_barriers=vb)
    bins = get_bins(events, close)
    bins = bins[bins["bin"] != 0]
    idx = bins.index
    X = feats.reindex(idx)
    y, ret, t1 = bins["bin"].astype(int), bins["ret"], bins["t1"]

    print(f"イベント数 {len(idx)}  特徴量 {X.shape[1]}（うち flow/vpin は tick 固有）")
    res, dsr, n_eff = evaluate(X, y, ret, t1, close)
    print(f"実効サンプル数 {n_eff}\n")

    print(f"=== ドルバー・パイプライン（実BTC/JPY {N_DAYS}日, 8設定の最良）===")
    print(f"OOS Sharpe 平均/標準偏差 : {res.mean_sharpe:.4f} / {res.std_sharpe:.4f}  "
          f"負パス {res.frac_negative:.0%}")
    print(f"DSR（真Sharpe>0確率）    : {dsr:.3f}   ［OHLCV足ベースライン: ≈0.30］")
    if dsr >= 0.95:
        print("→ 有意なエッジの兆候。感度分析・別期間で要追検証（過学習に警戒）")
    else:
        print("→ DSR<0.95＝有意エッジ無し。ただしドルバーは統計特性が良くlabel品質は向上。"
              "\n  次は別ホライズン/メタラベリング/インバランスバー/より長期間で継続探索。")


if __name__ == "__main__":
    main()
