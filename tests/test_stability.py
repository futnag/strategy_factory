"""非定常性計測子（時間減衰Sharpe・サブ期間安定性）の検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.stability import (
    pre_post_sharpe,
    subperiod_sharpes,
    time_decayed_sharpe,
)


def _series(vals):
    idx = pd.date_range("2016-07-31", periods=len(vals), freq="ME")
    return pd.Series(vals, index=idx)


def test_time_decay_emphasizes_recent():
    # 前半 -1、後半 +1（全体平均0）→ 短い半減期なら直近(+)が支配し正
    s = _series([-0.01] * 24 + [0.01] * 24)
    decayed = time_decayed_sharpe(s, halflife=6.0)
    assert decayed > 0.5
    # 半減期を非常に長くすると等加重に近づき ≈0
    flat = time_decayed_sharpe(s, halflife=1e6)
    assert abs(flat) < 0.05


def test_time_decay_short_series_nan():
    assert np.isnan(time_decayed_sharpe(_series([0.01])))


def test_subperiod_sharpes_partition():
    s = _series(list(np.linspace(-0.02, 0.02, 30)))
    parts = subperiod_sharpes(s, k=3)
    assert len(parts) == 3
    assert sum(n for _, n, _ in parts) == 30          # 全数を分割
    # 各ラベルは YYYY-MM..YYYY-MM 形式
    assert all(".." in lbl for lbl, _, _ in parts)


def test_pre_post_split():
    s = _series([0.01] * 48)
    (n_pre, _), (n_post, _) = pre_post_sharpe(s, "2018-07-31")
    assert n_pre == 24 and n_post == 24                # 2016-07..2018-06 / 以降


def test_pre_post_sign():
    # 節目前は負、後は正
    s = _series([-0.02] * 12 + [0.02] * 12)
    (_, sh_pre), (_, sh_post) = pre_post_sharpe(s, "2017-07-31")
    assert sh_pre < 0 < sh_post
