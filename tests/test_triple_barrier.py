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


# --- 悲観モード（バー内 H/L・同足は損切り優先・約定規約）---------------------
def _hlc(close_vals, high_vals, low_vals):
    c = _close(close_vals)
    h = pd.Series(high_vals, index=c.index, dtype=float)
    lo = pd.Series(low_vals, index=c.index, dtype=float)
    return c, h, lo


def _one_event(close, **kw):
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt,
                        vertical_barriers=vb, **kw)
    return events, get_bins(events, close)


def test_pessimistic_same_bar_double_touch_prefers_stop():
    # idx1 のバー内で +2%/-2% の両バリアを掃き、引けはフラット。
    # 終値パス（既定）では無接触＝垂直、悲観モードでは「先に損切り」と判定する。
    close, high, low = _hlc([100, 100, 100, 100, 100, 100, 100],
                            [100, 103, 100, 100, 100, 100, 100],
                            [100, 97, 100, 100, 100, 100, 100])
    idx = close.index
    ev_close, bins_close = _one_event(close)
    assert ev_close.loc[idx[0], "t1"] == idx[5]          # 従来：垂直バリア
    ev, bins = _one_event(close, high=high, low=low)
    assert ev.loc[idx[0], "t1"] == idx[1]
    assert ev.loc[idx[0], "touch"] == "sl"               # 両接触 → 損切り優先
    assert bins.loc[idx[0], "ret"] == pytest.approx(-0.02)  # 水準と終値(0%)の悪い方
    assert bins.loc[idx[0], "bin"] == -1


def test_pessimistic_gap_through_stop_fills_at_close():
    # idx1 で -2% の stop を大きく飛び越えて暴落（low=90, close=92）→ 実勢で約定。
    close, high, low = _hlc([100, 92, 92, 92, 92, 92, 92],
                            [100, 100, 92, 92, 92, 92, 92],
                            [100, 90, 92, 92, 92, 92, 92])
    idx = close.index
    ev, bins = _one_event(close, high=high, low=low)
    assert ev.loc[idx[0], "touch"] == "sl"
    assert bins.loc[idx[0], "ret"] == pytest.approx(-0.08)   # min(-0.02, -0.08)


def test_pessimistic_pt_fills_at_barrier_level():
    # idx1 で高値が +2% バリアに接触（high=105）し引け +4%（close=104）→ 指値は水準で約定。
    close, high, low = _hlc([100, 104, 104, 104, 104, 104, 104],
                            [100, 105, 104, 104, 104, 104, 104],
                            [100, 100, 104, 104, 104, 104, 104])
    idx = close.index
    ev, bins = _one_event(close, high=high, low=low)
    assert ev.loc[idx[0], "touch"] == "pt"
    assert bins.loc[idx[0], "ret"] == pytest.approx(0.02)    # +4% でなくバリア水準
    assert bins.loc[idx[0], "bin"] == 1


def test_pessimistic_entry_bar_extremes_ignored():
    # エントリーバー idx0 の高値はバリア超だが、建値=当バー終値なので接触に使わない。
    close, high, low = _hlc([100, 100, 100, 100, 100, 100, 100],
                            [110, 100, 100, 100, 100, 100, 100],
                            [95, 100, 100, 100, 100, 100, 100])
    idx = close.index
    ev, bins = _one_event(close, high=high, low=low)
    assert ev.loc[idx[0], "touch"] == "t1"               # 垂直のみ
    assert ev.loc[idx[0], "t1"] == idx[5]


def test_pessimistic_short_side_uses_high_as_adverse():
    # short(side=-1)：不利方向は高値。idx1 high=103 で -2%（side調整後）の stop に接触。
    close, high, low = _hlc([100, 102, 102, 102, 102, 102, 102],
                            [100, 103, 102, 102, 102, 102, 102],
                            [100, 101, 102, 102, 102, 102, 102])
    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.02, index=t_events)
    side = pd.Series([-1.0], index=t_events)
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    events = get_events(close, t_events, pt_sl=[1, 1], trgt=trgt,
                        vertical_barriers=vb, side=side, high=high, low=low)
    bins = get_bins(events, close)
    assert events.loc[idx[0], "touch"] == "sl"
    assert bins.loc[idx[0], "ret"] == pytest.approx(-0.02)   # min(-0.02, -0.02)
    assert bins.loc[idx[0], "bin"] == 0                      # メタ：負け


def test_pessimistic_requires_both_high_and_low():
    close = _close([100, 100, 100])
    t_events = pd.DatetimeIndex([close.index[0]])
    trgt = pd.Series(0.02, index=t_events)
    with pytest.raises(ValueError):
        get_events(close, t_events, pt_sl=[1, 1], trgt=trgt, high=close)
