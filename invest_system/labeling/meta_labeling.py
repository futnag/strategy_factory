"""メタラベリング（補正的AI）とベットサイジング（L7）。

統合ナレッジベース §4.2 / DP4。AFML (López de Prado 2018) ch.3.6 / ch.10。

二段構造：
  一次モデル … 「方向（side, long=+1/short=-1）」を決める（ルールでもMLでも可）。
  二次（メタ）モデル … 「その賭けに乗るか・サイズ」を決める。get_bins(side=...) の
                       {0,1} ラベル（賭けが当たったか）で学習し、予測確率からサイズを算出。

メタラベリングは Precision を高め、偽陽性（空振り）を排除する。高Recall・低Precision
な一次モデルを救済できる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .triple_barrier import get_bins, get_events


def bet_size_from_prob(prob, num_classes: int = 2):
    """予測確率からベットサイズを算出。AFML snippet 10.1。

    z = (p − 1/num_classes) / sqrt(p(1−p));  size = 2Φ(z) − 1 ∈ [−1, 1]。
    無情報 p=1/num_classes で 0、確信が高いほど ±1 に近づく。
    """
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1.0 - 1e-9)
    z = (p - 1.0 / num_classes) / np.sqrt(p * (1.0 - p))
    size = 2.0 * norm.cdf(z) - 1.0
    return float(size) if np.ndim(prob) == 0 else size


def meta_position(side, meta_prob, num_classes: int = 2):
    """メタ確率からポジションを算出： side × max(0, size)。

    当たり確率 < 0.5 の賭けは見送る（サイズ 0）。side と meta_prob はスカラ/配列可。
    """
    size = np.clip(bet_size_from_prob(meta_prob, num_classes), 0.0, 1.0)
    return np.asarray(side, dtype=float) * size


def meta_labels(close: pd.Series, t_events: pd.DatetimeIndex, side: pd.Series,
                pt_sl, trgt: pd.Series, vertical_barriers: pd.Series,
                min_ret: float = 0.0) -> pd.DataFrame:
    """一次モデルの side を用いてメタラベル {0,1} と side 調整リターンを生成。

    get_events(side=...) + get_bins の薄いラッパ。Returns get_bins の DataFrame
    （bin ∈ {0,1}, ret は side 調整済み）。
    """
    events = get_events(close, t_events, pt_sl, trgt, min_ret=min_ret,
                        vertical_barriers=vertical_barriers, side=side)
    return get_bins(events, close)
