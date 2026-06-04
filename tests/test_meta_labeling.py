"""メタラベリングのベットサイジングとラベル生成を検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.labeling.meta_labeling import (
    bet_size_from_prob,
    meta_labels,
    meta_position,
)
from invest_system.labeling.triple_barrier import get_vertical_barriers


def test_bet_size_zero_at_no_information():
    assert bet_size_from_prob(0.5) == pytest.approx(0.0)


def test_bet_size_monotonic_and_bounded():
    assert bet_size_from_prob(0.9) > bet_size_from_prob(0.6) > 0.0
    assert bet_size_from_prob(0.1) < 0.0
    for p in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert -1.0 <= bet_size_from_prob(p) <= 1.0


def test_bet_size_vectorized():
    out = bet_size_from_prob(np.array([0.5, 0.9]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] > 0.0


def test_meta_position_skips_low_confidence():
    assert meta_position(1.0, 0.5) == pytest.approx(0.0)    # 無情報 → 不参加
    assert meta_position(1.0, 0.3) == pytest.approx(0.0)    # <0.5 → 見送り
    assert meta_position(1.0, 0.9) > 0.0                    # ロング・サイズ付き
    assert meta_position(-1.0, 0.9) < 0.0                   # ショート・サイズ付き


def test_meta_labels_are_binary_and_correct():
    # 価格下落 → short(side=-1) は当たり → bin=1
    close = pd.Series([100, 99, 97, 99, 100, 100, 100], dtype=float,
                      index=pd.date_range("2020-01-01", periods=7, freq="D"))
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    side = pd.Series([-1.0], index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    bins = meta_labels(close, t_events, side, [1, 1], trgt, vb)
    assert set(np.unique(bins["bin"])) <= {0.0, 1.0}
    assert bins.loc[idx[0], "bin"] == 1
