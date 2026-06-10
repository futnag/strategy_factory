"""時系列モメンタム（TSMOM）：サイン×ボラターゲットの古典形（柱E 候補）。

Moskowitz–Ooi–Pedersen (2012, JFE) の標準形。各資産を**自分の過去とだけ**比較する
（クロスセクション・モメンタム＝銘柄間の相対比較とは別の現象。日本株で棄却済みなのは
後者＝docs/03 §6.7）。シグナルは過去 L ヶ月リターンの符号のみ・ポジションは資産ごとの
リスクが揃うようボラの逆数でスケール：

    w_i(t) = sign( close_t / close_{t-L} − 1 ) × ( σ_target / σ_i(t) ) / N

σ_i は日次リターンの trailing 実現ボラ（年率・下限 floor でゼロ割りと過大レバを防ぐ）、
N は当日シグナルが有効な資産数。パラメータは実質 L と σ_target の2つ＝Chan の
「シンプルさの優位」。PIT：シグナルは月末終値（t 時点既知）、σ は ≤t の日次のみ。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_vol(daily_close: pd.DataFrame, *, window: int = 63,
                   floor: float = 0.05, ann: float = 252.0) -> pd.DataFrame:
    """日次終値 wide → trailing 実現ボラ（年率）。下限 floor でクリップ。

    祝日ずれ（資産ごとのカレンダー差）に頑健なよう、各列の有効値ベースで
    pct_change する（NaN は飛ばして直近の有効値と比較しない＝単純に NaN のまま）。
    """
    ret = daily_close.pct_change(fill_method=None)
    vol = ret.rolling(window, min_periods=max(20, window // 2)).std() * np.sqrt(ann)
    return vol.clip(lower=floor)


def tsmom_weights(monthly_close: pd.DataFrame, vol_asof: pd.DataFrame,
                  lookback: int, *, vol_target: float = 0.10) -> dict:
    """{決定日: ウェイト Series}。sign(過去 lookback ヶ月リターン) × (σ_tgt/σ)/N。

    monthly_close: index=月末決定日, col=資産（各行は t 時点で既知の終値）。
    vol_asof: 同形（決定日における年率σ、≤t 情報で構築済みのこと）。
    リターンが計算できない（履歴不足/NaN）資産・σ が NaN の資産はその日は無ポジ。
    sign==0（完全フラット）も無ポジ。N は当日有効な資産数。
    """
    mom = monthly_close / monthly_close.shift(lookback) - 1.0
    out: dict[pd.Timestamp, pd.Series] = {}
    for t in monthly_close.index:
        sig = np.sign(mom.loc[t])
        vol = vol_asof.loc[t] if t in vol_asof.index else pd.Series(dtype="float64")
        w = (sig * (vol_target / vol)).replace([np.inf, -np.inf], np.nan).dropna()
        w = w[w != 0.0]
        if len(w):
            out[t] = w / len(w)
    return out


def blend_weights(weight_sets: list[dict]) -> dict:
    """複数の {決定日: ウェイト} を等加重で合成（ルックバック分散の combo 用）。"""
    dates = sorted(set().union(*[set(ws) for ws in weight_sets]))
    out: dict[pd.Timestamp, pd.Series] = {}
    k = float(len(weight_sets))
    for t in dates:
        total: pd.Series | None = None
        for ws in weight_sets:
            w = ws.get(t)
            if w is None:
                continue
            total = w / k if total is None else total.add(w / k, fill_value=0.0)
        if total is not None and len(total[total != 0.0]):
            out[t] = total[total != 0.0]
    return out
