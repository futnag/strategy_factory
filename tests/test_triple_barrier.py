"""トリプルバリア法の第一接触・ラベル・メタラベリングを検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.labeling.triple_barrier import (
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)


def _close(values, start="2020-01-01", freq="D"):
    return pd.Series(values, dtype=float,
                     index=pd.date_range(start, periods=len(values), freq=freq),
                     name="close")


# --- ボラティリティ目標 ---------------------------------------------------
def test_get_vol_positive_and_aligned():
    rng = np.random.default_rng(0)
    close = _close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    vol = get_vol(close, span=20, lookback=1).dropna()
    assert len(vol) > 0
    assert (vol > 0).all()


# --- 垂直バリア -----------------------------------------------------------
def test_vertical_barriers_positions():
    close = _close(range(10))
    idx = close.index
    vb = get_vertical_barriers(close, idx[:5], num_bars=3)
    assert len(vb) == 5
    assert vb.loc[idx[0]] == idx[3]
    assert vb.loc[idx[4]] == idx[7]


def test_vertical_barriers_drop_tail():
    close = _close(range(10))
    idx = close.index
    vb = get_vertical_barriers(close, idx[:9], num_bars=3)
    assert len(vb) == 7              # pos 7,8 → 10,11 が範囲外で脱落
    assert vb.index[-1] == idx[6]


# --- 第一接触とラベル -----------------------------------------------------
def test_upper_barrier_touched_first():
    # 100 → +3% で上方バリア(102)を idx2 で突破
    close = _close([100, 101, 103, 102, 100, 100, 100])
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt, vertical_barriers=vb)
    bins = get_bins(events, close)
    assert events.loc[idx[0], "t1"] == idx[2]
    assert bins.loc[idx[0], "bin"] == 1
    assert bins.loc[idx[0], "ret"] == pytest.approx(0.03)


def test_lower_barrier_touched_first():
    # 100 → -3% で下方バリア(98)を idx2 で突破
    close = _close([100, 99, 97, 99, 100, 100, 100])
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt, vertical_barriers=vb)
    bins = get_bins(events, close)
    assert events.loc[idx[0], "t1"] == idx[2]
    assert bins.loc[idx[0], "bin"] == -1


def test_vertical_barrier_touched_first():
    # ±2% 以内に収まり、垂直バリア(idx3)で確定
    close = _close([100, 100.5, 101, 100.8, 100, 100])
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=3)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt, vertical_barriers=vb)
    bins = get_bins(events, close)
    assert events.loc[idx[0], "t1"] == idx[3]        # 水平バリア未接触 → 垂直で確定
    assert bins.loc[idx[0], "bin"] == 1              # 終値 +0.8% → sign +1


# --- メタラベリング -------------------------------------------------------
def test_meta_labeling_short_bet_wins():
    # 価格下落。short(side=-1) の賭けは利益 → bin=1
    close = _close([100, 99, 97, 99, 100, 100, 100])
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    side = pd.Series([-1.0], index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt,
                        vertical_barriers=vb, side=side)
    bins = get_bins(events, close)
    assert set(np.unique(bins["bin"])) <= {0.0, 1.0}   # メタラベルは {0,1}
    assert bins.loc[idx[0], "bin"] == 1                # short が的中
    assert bins.loc[idx[0], "ret"] > 0


def test_meta_labeling_long_bet_loses():
    # 価格下落。long(side=+1) の賭けは損失 → bin=0
    close = _close([100, 99, 97, 99, 100, 100, 100])
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    side = pd.Series([1.0], index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt,
                        vertical_barriers=vb, side=side)
    bins = get_bins(events, close)
    assert bins.loc[idx[0], "bin"] == 0
    assert bins.loc[idx[0], "ret"] < 0


def test_min_ret_filters_events():
    close = _close([100, 101, 102, 103, 104, 105])
    idx = close.index
    t_events = idx[:3]
    trgt = pd.Series(0.001, index=t_events)   # 全イベントの幅が min_ret 未満
    vb = get_vertical_barriers(close, t_events, num_bars=2)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt,
                        min_ret=0.01, vertical_barriers=vb)
    assert events.empty
