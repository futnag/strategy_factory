"""非定常性への対処：時間減衰Sharpe と サブ期間安定性。

「市場は10年前と違う」への実務的回答は『長期データを使うが旧い部分を割り引き、
サブ期間で安定性を検定する』こと。ここではその2つの計測子を提供する：
  - time_decayed_sharpe : 直近を重く指数減衰した per-period Sharpe
  - subperiod_sharpes   : 期間を等分し各期の年率Sharpeを出す（レジーム依存の可視化）
  - pre_post_sharpe     : 構造節目（例 2020）の前後でSharpeRを対比
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANN = np.sqrt(12.0)  # 月次→年率


def _ann_sharpe(x: pd.Series) -> float:
    if len(x) >= 2 and x.std(ddof=1) > 0:
        return float(x.mean() / x.std(ddof=1) * _ANN)
    return float("nan")


def time_decayed_sharpe(returns: pd.Series, halflife: float = 36.0) -> float:
    """指数時間減衰を施した per-period Sharpe（直近ほど重い）。

    halflife（月）ごとに重みが半減。直近の観測が age=0 で重み1。
    非定常下で「いま効いているか」を全標本を捨てずに測る。
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = r.size
    if n < 2:
        return float("nan")
    age = np.arange(n)[::-1]              # 末尾(最新)=0, 先頭(最古)=n-1
    w = 0.5 ** (age / float(halflife))
    w = w / w.sum()
    mu = float((w * r).sum())
    var = float((w * (r - mu) ** 2).sum())
    if var <= 0:
        return float("nan")
    return mu / np.sqrt(var)


def subperiod_sharpes(returns: pd.Series, k: int = 3) -> list[tuple[str, int, float]]:
    """系列を時間順に k 等分し、各サブ期間の (ラベル, n, 年率Sharpe) を返す。"""
    r = returns.dropna()
    if len(r) < k or k < 1:
        return []
    out: list[tuple[str, int, float]] = []
    for idx in np.array_split(np.arange(len(r)), k):
        sub = r.iloc[idx]
        label = f"{sub.index[0]:%Y-%m}..{sub.index[-1]:%Y-%m}"
        out.append((label, int(len(sub)), _ann_sharpe(sub)))
    return out


def pre_post_sharpe(returns: pd.Series, split: str | pd.Timestamp
                    ) -> tuple[tuple[int, float], tuple[int, float]]:
    """構造節目 split の前後で (n, 年率Sharpe) を対比。"""
    r = returns.dropna()
    s = pd.Timestamp(split)
    pre, post = r[r.index < s], r[r.index >= s]
    return (len(pre), _ann_sharpe(pre)), (len(post), _ann_sharpe(post))
