"""サンプル独自性と逐次ブートストラップ（L6）。

統合ナレッジベース §4.3 / DP5 の実装。AFML (López de Prado 2018) ch.4。

金融ラベルは [t0, t1] の期間を持ち、隣接ラベルは期間が重なるため独立でない
（非IID）。重なりを「同時イベント数」で測り、各ラベルの「平均独自性」を算出。
これに基づく重みづけ・逐次ブートストラップでアウトオブサンプル精度を高める。

イベントは pandas.Series ``t1``：index = ラベル開始 t0, value = ラベル終了 t1。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def num_concurrent_events(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """各バーで同時にアクティブなラベル数。AFML snippet 4.1。"""
    t1 = t1.dropna()
    if t1.empty:
        return pd.Series(dtype=float)
    span = bar_index[(bar_index >= t1.index[0]) & (bar_index <= t1.max())]
    count = pd.Series(0, index=span)
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1
    return count


def average_uniqueness(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """各ラベルの平均独自性 = ラベル期間にわたる 1/同時数 の平均。AFML snippet 4.2。

    重なりの多いラベルほど独自性が低い（→学習で重みを下げるべき）。
    """
    t1 = t1.dropna()
    conc = num_concurrent_events(bar_index, t1)
    uniq = pd.Series(index=t1.index, dtype=float)
    for t_in, t_out in t1.items():
        uniq.loc[t_in] = (1.0 / conc.loc[t_in:t_out]).mean()
    return uniq


def sample_weights_by_return(bar_index: pd.DatetimeIndex, t1: pd.Series,
                             close: pd.Series, normalize: bool = True) -> pd.Series:
    """リターン帰属に基づくサンプル重み。AFML snippet 4.10。

    各ラベルに、保有期間の対数収益を同時数で割って帰属させた絶対値を重みとする。
    normalize=True で重みの総和をサンプル数 N に正規化（平均1）。
    """
    t1 = t1.dropna()
    conc = num_concurrent_events(bar_index, t1)
    log_ret = np.log(close).diff()
    w = pd.Series(index=t1.index, dtype=float)
    for t_in, t_out in t1.items():
        w.loc[t_in] = (log_ret.loc[t_in:t_out] / conc.loc[t_in:t_out]).sum()
    w = w.abs()
    if normalize and w.sum() > 0:
        w = w * len(w) / w.sum()
    return w


def time_decay(av_uniqueness: pd.Series, last_weight: float = 1.0) -> pd.Series:
    """時間減衰：累積独自性に対し線形に重みを減衰させる。AFML snippet 4.11。

    last_weight=1 で減衰無し（全て1）。0 で最古が0。負で一部の古い観測を0にする。
    """
    clf = av_uniqueness.sort_index().cumsum()
    if clf.iloc[-1] == 0:
        return pd.Series(1.0, index=av_uniqueness.index)
    if last_weight >= 0:
        slope = (1.0 - last_weight) / clf.iloc[-1]
    else:
        slope = 1.0 / ((last_weight + 1) * clf.iloc[-1])
    const = 1.0 - slope * clf.iloc[-1]
    decay = const + slope * clf
    decay[decay < 0] = 0.0
    return decay


# --- 指標行列ベース：逐次ブートストラップ ---------------------------------
def get_indicator_matrix(bar_index: pd.DatetimeIndex, t1: pd.Series) -> pd.DataFrame:
    """指標行列（バー×イベント）。イベント j がバー i でアクティブなら 1。AFML 4.3。"""
    t1 = t1.dropna()
    span = bar_index[(bar_index >= t1.index[0]) & (bar_index <= t1.max())]
    ind = pd.DataFrame(0.0, index=span, columns=np.arange(len(t1)))
    for j, (t_in, t_out) in enumerate(t1.items()):
        ind.loc[t_in:t_out, j] = 1.0
    return ind


def average_uniqueness_from_indicator(ind_mat: pd.DataFrame) -> pd.Series:
    """指標行列から各イベントの平均独自性を算出。AFML snippet 4.4。"""
    conc = ind_mat.sum(axis=1)
    out = pd.Series(index=ind_mat.columns, dtype=float)
    for col in ind_mat.columns:
        active = ind_mat[col] > 0
        out[col] = float((1.0 / conc[active]).mean()) if active.any() else 0.0
    return out


def sequential_bootstrap(ind_mat: pd.DataFrame, size: Optional[int] = None,
                         random_state: Optional[int] = None) -> list:
    """逐次ブートストラップ：独自性が高い標本を優先的に抽出。AFML snippet 4.5。

    すでに抽出した標本との重なりが少ない（独自性が高い）候補ほど選ばれやすくし、
    標準ブートストラップより冗長性の低い（独立性の高い）標本集合を得る。
    返り値：抽出したイベント列インデックスのリスト（重複あり）。
    """
    rng = np.random.default_rng(random_state)
    cols = list(ind_mat.columns)
    if size is None:
        size = len(cols)
    phi: list[int] = []
    while len(phi) < size:
        avg_u = pd.Series(0.0, index=cols)
        for col in cols:
            sub = ind_mat[phi + [col]]
            conc = sub.sum(axis=1)
            active = ind_mat[col] > 0
            avg_u[col] = float((1.0 / conc[active]).mean())
        prob = (avg_u / avg_u.sum()).to_numpy()
        phi.append(int(rng.choice(cols, p=prob)))
    return phi
