"""トリプルバリア法（L5）。

統合ナレッジベース §4.1 / DP4 の実装。AFML (López de Prado 2018) ch.3。

「n日後のリターン」という固定ホライズン法はパス依存性（途中のストップアウト）を
無視する。トリプルバリア法は3つの境界の「最初に触れたもの」をラベルとする：
  1. 上方水平バリア（利確, profit-taking）
  2. 下方水平バリア（損切, stop-loss）
  3. 垂直バリア（時間切れ, time-out）

``close`` は DatetimeIndex（昇順・一意）の価格 Series を想定。

バー内の解像度（intrabar）について：既定（high/low 省略時）は AFML の教科書どおり
**終値パス**でバリア接触を判定する＝バー内で H/L がどちらの順に発生したかという
ブラックボックスに依存しない一方、バー内でバリアを貫通して引けまでに戻った接触は
見えない（実ストップ注文より楽観）。`get_events(..., high=, low=)` を与えると
**悲観モード**になる：接触はバー内の H/L で判定し、同一バーで利確・損切り双方に
触れた場合は**必ず損切りが先**とみなす（ワーストケース・発生順序の楽観を排除）。
エントリーバー自体の H/L は判定から除外する（建値は当バー終値＝それ以前の高安では
約定し得ないため）。約定リターンは get_bins が touch 種別で決める：利確=バリア水準
（指値はそれ以上有利にならない）、損切り=**バリア水準と当該バー終値の悪い方**
（ギャップで stop を飛び越えた場合は実勢で約定）、時間切れ=終値。
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
                      pt_sl: Sequence[float],
                      high: Optional[pd.Series] = None,
                      low: Optional[pd.Series] = None) -> pd.DataFrame:
    """各イベントについて、[t0, t1] 内で上方/下方水平バリアに最初に触れた時刻を求める。

    AFML snippet 3.2。返り値の列：t1（垂直）, pt（上方接触時刻）, sl（下方接触時刻）。
    ``events`` は列 t1, trgt, side を持つ。side はメタラベリングの方向（long=+1/short=-1）。

    high/low を与えると**悲観モード**：接触をバー内 H/L（side 調整後の有利方向=high系・
    不利方向=low系）で判定する。エントリーバーの H/L は除外（建値=当バー終値）。
    同一バーで両接触した場合の優先順位は get_events 側で損切り優先に解決する。
    """
    out = events[["t1"]].copy()
    trgt = events["trgt"]
    pt = pt_sl[0] * trgt if pt_sl[0] > 0 else pd.Series(index=events.index, dtype=float)
    sl = -pt_sl[1] * trgt if pt_sl[1] > 0 else pd.Series(index=events.index, dtype=float)

    last = close.index[-1]
    for loc, t1 in events["t1"].fillna(last).items():
        side = events.at[loc, "side"]
        px0 = close.loc[loc]
        if high is not None and low is not None:
            fav = (high if side > 0 else low).loc[loc:t1].iloc[1:]   # 有利方向の極値
            adv = (low if side > 0 else high).loc[loc:t1].iloc[1:]   # 不利方向の極値
            fav_ret = (fav / px0 - 1.0) * side
            adv_ret = (adv / px0 - 1.0) * side
            out.at[loc, "sl"] = adv_ret[adv_ret < sl.loc[loc]].index.min()
            out.at[loc, "pt"] = fav_ret[fav_ret > pt.loc[loc]].index.min()
        else:
            returns = (close.loc[loc:t1] / px0 - 1.0) * side
            out.at[loc, "sl"] = returns[returns < sl.loc[loc]].index.min()  # 損切到達
            out.at[loc, "pt"] = returns[returns > pt.loc[loc]].index.min()  # 利確到達
    return out


def get_events(close: pd.Series, t_events: pd.DatetimeIndex,
               pt_sl: Sequence[float], trgt: pd.Series, min_ret: float = 0.0,
               vertical_barriers: Optional[pd.Series] = None,
               side: Optional[pd.Series] = None,
               high: Optional[pd.Series] = None,
               low: Optional[pd.Series] = None) -> pd.DataFrame:
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
    high / low : pd.Series  与えると**悲観モード**（モジュール docstring 参照）：接触は
        バー内 H/L で判定し、同一バーの両接触は損切り優先。touch/pt_ret/sl_ret 列を追加
        で返し、get_bins が約定規約（利確=水準・損切り=水準と終値の悪い方）に使う。

    Returns
    -------
    events : DataFrame  列 [t1（第一接触時刻）, trgt(, side)(, touch, pt_ret, sl_ret)]。
    """
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    if trgt.empty:
        return pd.DataFrame(columns=["t1", "trgt"])

    if (high is None) != (low is None):
        raise ValueError("high/low は両方与えるか両方省略（悲観モードの片翼は不可）")

    if vertical_barriers is None:
        vertical_barriers = pd.Series(pd.NaT, index=t_events)
    side_ = (pd.Series(1.0, index=trgt.index) if side is None
             else side.reindex(trgt.index))

    events = pd.concat(
        {"t1": vertical_barriers.reindex(trgt.index), "trgt": trgt, "side": side_},
        axis=1).dropna(subset=["trgt", "side"])

    touches = apply_pt_sl_on_t1(close, events, pt_sl, high=high, low=low)
    first = touches[["t1", "pt", "sl"]].min(axis=1)         # 最初に触れた境界
    if high is not None:
        # 悲観モード：接触種別を記録（同時刻なら sl 優先＝ワーストケース）し、
        # 約定規約に使う side 調整後のバリア水準を持たせる。
        sl_first = touches["sl"].notna() & (touches["sl"] <= first)
        pt_first = touches["pt"].notna() & (touches["pt"] <= first) & ~sl_first
        events["touch"] = np.where(sl_first, "sl",
                                   np.where(pt_first, "pt", "t1"))
        events["pt_ret"] = (pt_sl[0] * events["trgt"]
                            if pt_sl[0] > 0 else np.nan)
        events["sl_ret"] = (-pt_sl[1] * events["trgt"]
                            if pt_sl[1] > 0 else np.nan)
    events["t1"] = first
    if side is None:
        events = events.drop(columns="side")
    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """第一接触から実現リターンとラベルを計算。AFML snippet 3.5／3.7。

    side 無し（一次）：bin = sign(ret) ∈ {-1,0,+1}。
    side 有り（メタ）：bin ∈ {0,1}（その賭けが利益を生んだか）。

    events に touch 列があれば（get_events の悲観モード）約定規約を適用する：
      pt → pt_ret（指値はバリア水準で約定＝それ以上有利にならない）
      sl → min(sl_ret, 接触バー終値リターン)（ギャップで stop を飛び越えたら実勢約定）
      t1 → 終値リターン（従来どおり）
    いずれも side 調整後のリターン空間で計算する。
    """
    events_ = events.dropna(subset=["t1"])
    # バリア時刻・イベント時刻はいずれも close.index の要素である前提（厳密一致で参照）
    px0 = close.loc[events_.index].to_numpy()
    px1 = close.loc[events_["t1"].to_numpy()].to_numpy()

    out = pd.DataFrame(index=events_.index)
    out["ret"] = px1 / px0 - 1.0
    if "side" in events_.columns:
        out["ret"] *= events_["side"].to_numpy()
    if "touch" in events_.columns:              # 悲観モードの約定規約
        touch = events_["touch"].to_numpy()
        close_ret = out["ret"].to_numpy()       # side 調整済みの終値リターン
        ret = close_ret.copy()
        is_pt = touch == "pt"
        is_sl = touch == "sl"
        ret[is_pt] = events_["pt_ret"].to_numpy()[is_pt]
        ret[is_sl] = np.minimum(events_["sl_ret"].to_numpy()[is_sl],
                                close_ret[is_sl])
        out["ret"] = ret
        out["touch"] = touch
    out["bin"] = np.sign(out["ret"])
    if "side" in events_.columns:
        out.loc[out["ret"] <= 0, "bin"] = 0.0  # メタラベリング：{0,1}
    out["t1"] = events_["t1"].to_numpy()
    return out
