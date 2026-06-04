"""トリプルバリア法（L5）。

統合ナレッジベース §4.1 / DP4 の実装。AFML (López de Prado 2018) ch.3。

「n日後のリターン」という固定ホライズン法はパス依存性（途中のストップアウト）を
無視する。トリプルバリア法は3つの境界の「最初に触れたもの」をラベルとする：
  1. 上方水平バリア（利確, profit-taking）
  2. 下方水平バリア（損切, stop-loss）
  3. 垂直バリア（時間切れ, time-out）

``close`` は DatetimeIndex（昇順・一意）の価格 Series を想定。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def get_vol(close: pd.Series, span: int = 100, lookback: int = 1) -> pd.Series:
    """動的ボラティリティ目標（バリア幅）。lookback バー収益の EWMA 標準偏差。

    AFML snippet 3.1 を一般化（日次でなくバー単位）。先頭 lookback 件は NaN。
    """
    rets = close.pct_change(lookback, fill_method=None)
    return rets.ewm(span=span).std()


def get_vertical_barriers(close: pd.Series, t_events: pd.DatetimeIndex,
                          num_bars: int) -> pd.Series:
    """各イベントの垂直バリア（num_bars 本先の時刻）。AFML snippet 3.4。

    系列末尾を超えるイベントは脱落させる。返り値 index = 有効な t_events。
    """
    if num_bars <= 0:
        raise ValueError("num_bars must be > 0")
    pos = close.index.searchsorted(t_events) + num_bars
    valid = pos < close.shape[0]
    return pd.Series(close.index[pos[valid]], index=t_events[valid])


def apply_pt_sl_on_t1(close: pd.Series, events: pd.DataFrame,
                      pt_sl: Sequence[float]) -> pd.DataFrame:
    """各イベントについて、[t0, t1] 内で上方/下方水平バリアに最初に触れた時刻を求める。

    AFML snippet 3.2。返り値の列：t1（垂直）, pt（上方接触時刻）, sl（下方接触時刻）。
    ``events`` は列 t1, trgt, side を持つ。side はメタラベリングの方向（long=+1/short=-1）。
    """
    out = events[["t1"]].copy()
    trgt = events["trgt"]
    pt = pt_sl[0] * trgt if pt_sl[0] > 0 else pd.Series(index=events.index, dtype=float)
    sl = -pt_sl[1] * trgt if pt_sl[1] > 0 else pd.Series(index=events.index, dtype=float)

    last = close.index[-1]
    for loc, t1 in events["t1"].fillna(last).items():
        path = close.loc[loc:t1]
        returns = (path / close.loc[loc] - 1.0) * events.at[loc, "side"]
        out.at[loc, "sl"] = returns[returns < sl.loc[loc]].index.min()  # 損切到達
        out.at[loc, "pt"] = returns[returns > pt.loc[loc]].index.min()  # 利確到達
    return out


def get_events(close: pd.Series, t_events: pd.DatetimeIndex,
               pt_sl: Sequence[float], trgt: pd.Series, min_ret: float = 0.0,
               vertical_barriers: Optional[pd.Series] = None,
               side: Optional[pd.Series] = None) -> pd.DataFrame:
    """トリプルバリアのイベント表を作る。AFML snippet 3.3／3.6。

    Parameters
    ----------
    close : pd.Series       価格系列。
    t_events : DatetimeIndex  ラベル付け対象のイベント開始時刻。
    pt_sl : (float, float)  [上方乗数, 下方乗数]。0 なら当該バリア無効。
    trgt : pd.Series        各イベントのバリア幅（get_vol 等）。
    min_ret : float         この幅未満のイベントは捨てる。
    vertical_barriers : pd.Series  各イベントの時間切れ時刻（推奨）。None なら無し。
    side : pd.Series        メタラベリング用の方向（long=+1/short=-1）。None なら一次ラベル。

    Returns
    -------
    events : DataFrame  列 [t1（第一接触時刻）, trgt(, side)]。
    """
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    if trgt.empty:
        return pd.DataFrame(columns=["t1", "trgt"])

    if vertical_barriers is None:
        vertical_barriers = pd.Series(pd.NaT, index=t_events)
    side_ = (pd.Series(1.0, index=trgt.index) if side is None
             else side.reindex(trgt.index))

    events = pd.concat(
        {"t1": vertical_barriers.reindex(trgt.index), "trgt": trgt, "side": side_},
        axis=1).dropna(subset=["trgt", "side"])

    touches = apply_pt_sl_on_t1(close, events, pt_sl)
    events["t1"] = touches[["t1", "pt", "sl"]].min(axis=1)  # 最初に触れた境界
    if side is None:
        events = events.drop(columns="side")
    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """第一接触から実現リターンとラベルを計算。AFML snippet 3.5／3.7。

    side 無し（一次）：bin = sign(ret) ∈ {-1,0,+1}。
    side 有り（メタ）：bin ∈ {0,1}（その賭けが利益を生んだか）。
    """
    events_ = events.dropna(subset=["t1"])
    # バリア時刻・イベント時刻はいずれも close.index の要素である前提（厳密一致で参照）
    px0 = close.loc[events_.index].to_numpy()
    px1 = close.loc[events_["t1"].to_numpy()].to_numpy()

    out = pd.DataFrame(index=events_.index)
    out["ret"] = px1 / px0 - 1.0
    if "side" in events_.columns:
        out["ret"] *= events_["side"].to_numpy()
    out["bin"] = np.sign(out["ret"])
    if "side" in events_.columns:
        out.loc[out["ret"] <= 0, "bin"] = 0.0  # メタラベリング：{0,1}
    out["t1"] = events_["t1"].to_numpy()
    return out
