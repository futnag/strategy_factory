"""市場レジームのラベラ（柱D・Regime-Switching の土台・KB §11 / §8-3）。

トレンド強度（Kaufman Efficiency Ratio）とボラティリティを、**拡張窓 percentile の
三分位**でレジーム化する純関数群。閾値を全期間分位でなく ≤t 分布から取るのが先読み
回避の肝（feature_store.build_regime と同方式）。すべて「渡された ≤t 系列/パネル」で
因果計算＝先読みなし（DP12）。RegimeGated（research）に PIT 系列として供給する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_market(close: pd.DataFrame) -> pd.Series:
    """等加重マーケット水準（日次リターン平均の累積）。≤t のみで定まる（PIT）。"""
    mkt_ret = close.pct_change().mean(axis=1).fillna(0.0)
    return (1.0 + mkt_ret).cumprod()


def efficiency_ratio(level: pd.Series, window: int = 60) -> pd.Series:
    """Kaufman Efficiency Ratio = |ΔP(window)| / Σ|ΔP(1)|（窓内）。

    1=純トレンド（一方向）・0=純レンジ（往復で相殺）。有界 [0,1]・OHLC 不要・頑健。
    direction（正味移動）/ volatility（経路長）。≤t のみ参照＝先読みなし。
    """
    direction = level.diff(window).abs()
    volatility = level.diff().abs().rolling(window, min_periods=window).sum()
    return direction / volatility.replace(0.0, np.nan)


def expanding_tertile(s: pd.Series, min_periods: int = 252,
                      labels=(0, 1, 2)) -> pd.Series:
    """拡張窓 percentile の三分位ラベル（0/1/2）。閾値を ≤t 分布から取る＝PIT。

    s[t] の「≤t での percentile rank」を三分位に割る。全期間分位を使わないので
    将来情報が閾値に混入しない（build_regime と同方式）。min_periods 未満は NaN。
    """
    pct = s.expanding(min_periods=min_periods).rank(pct=True)
    return pd.cut(pct, [0.0, 1 / 3, 2 / 3, 1.0], labels=list(labels),
                  include_lowest=True).astype("float64")


def trend_regime(close: pd.DataFrame, window: int = 60,
                 min_periods: int = 252) -> pd.Series:
    """市場トレンド強度レジーム：0=レンジ(MR有利)・1=中・2=強トレンド(不利)。

    等加重マーケット水準の Efficiency Ratio を拡張窓三分位化。強トレンド/急変局面
    （共和分崩壊が起きやすい）を 2 として識別し、MR を 0（レンジ）に寄せる材料。
    """
    er = efficiency_ratio(equal_weight_market(close), window)
    return expanding_tertile(er, min_periods)


def vol_regime(close: pd.DataFrame, window: int = 60,
               min_periods: int = 252) -> pd.Series:
    """市場ボラティリティレジーム：0=低・1=中・2=高(不利)。build_regime と同規約。

    等加重マーケット日次リターンの実現ボラ（年率）を拡張窓三分位化。高ボラ＝相関崩壊・
    スプレッド発散の局面（research_price_factors の vol_regime<2 と同じ向き）。
    """
    mkt_ret = close.pct_change().mean(axis=1)
    rv = mkt_ret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252.0)
    return expanding_tertile(rv, min_periods)
