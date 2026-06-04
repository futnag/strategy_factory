"""マイクロ構造・流動性特徴（実エッジ探索）。

統合ナレッジベース §3.2 / §8。OHLCV バーのみから計算できる、López de Prado が
重視するマイクロ構造・流動性の特徴量群。tick データ不要で実データに適用可能。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_LN2 = np.log(2.0)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson ボラティリティ推定（高安レンジ）。"""
    rng2 = np.log(high / low) ** 2
    return np.sqrt((1.0 / (4.0 * _LN2)) * rng2.rolling(window).mean())


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, window: int = 20) -> pd.Series:
    """Garman-Klass ボラティリティ推定（OHLC）。"""
    hl = np.log(high / low) ** 2
    co = np.log(close / open_) ** 2
    gk = 0.5 * hl - (2.0 * _LN2 - 1.0) * co
    return np.sqrt(gk.rolling(window).mean().clip(lower=0.0))


def amihud_illiquidity(close: pd.Series, dollar_volume: pd.Series,
                       window: int = 20) -> pd.Series:
    """Amihud 非流動性 = |リターン| / 取引代金 の窓平均。高いほど非流動的。"""
    ret = close.pct_change(fill_method=None).abs()
    illiq = ret / dollar_volume.replace(0.0, np.nan)
    return illiq.rolling(window).mean()


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll の実効スプレッド推定：連続価格変化の系列共分散から。"""
    dp = close.diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    return 2.0 * np.sqrt((-cov).clip(lower=0.0))


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin-Schultz 高安スプレッド推定（2012）。"""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    gamma = np.log(h2 / l2) ** 2
    den = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0)


def vpin(close: pd.Series, volume: pd.Series, window: int = 50,
         sigma_span: int = 50) -> pd.Series:
    """VPIN（情報トレーダー確率）の OHLCV 近似。値域 [0,1]。

    Bulk Volume Classification：買い出来高比率 = Φ(ΔP/σ) で出来高を売買に按分し、
    不均衡 |V_buy − V_sell| を窓で集計。Easley, López de Prado, O'Hara。
    """
    dp = close.diff()
    sigma = dp.ewm(span=sigma_span).std()
    z = (dp / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    buy_frac = pd.Series(norm.cdf(z.to_numpy()), index=close.index)
    imbalance = volume * (2.0 * buy_frac - 1.0).abs()
    return imbalance.rolling(window).sum() / volume.rolling(window).sum()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI（相対力指数, モメンタム振動子）。値域 [0,100]。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss != 0.0, 100.0)        # loss=0（全上昇）→ 100
