"""検証ファクトリ Phase1：バックテスト・エンジンの検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.data_view import AsOfView
from invest_system.research.engine import (
    apply_rebalance_band, backtest, _ann_factor, open_fill_backtest,
)
from invest_system.research.strategy import (
    CrossSectionalStrategy, GapReversal, SignalTimingStrategy,
)


def test_gap_engine_realizes_next_period_return():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    op = pd.DataFrame({"A": [100., 100, 88, 99], "B": [100., 100, 100, 101]}, index=idx)
    cl = pd.DataFrame({"A": [100., 100, 90, 99], "B": [100., 100, 101, 101]}, index=idx)
    view = AsOfView({"open": op, "close": cl})
    res = backtest(GapReversal(threshold=0.10, hold=1), view, costs_bps=15.0)
    # d2 で A 建玉 → d2→d3 リターン 99/90-1=0.10、回転1で 15bps 控除
    assert res.gross.loc[idx[2]] == pytest.approx(0.10)
    assert res.returns.loc[idx[2]] == pytest.approx(0.10 - 0.0015)
    assert res.returns.loc[idx[0]] == 0.0 and res.returns.loc[idx[1]] == 0.0
    assert res.n_positions.loc[idx[2]] == 1


def test_cross_sectional_engine_long_short_return():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    cl = pd.DataFrame({"A": [100., 95], "B": [100., 100], "C": [100., 100],
                       "D": [100., 100], "E": [100., 110]}, index=idx)
    factor = pd.DataFrame({"A": [1., 1], "B": [2., 2], "C": [3., 3],
                           "D": [4., 4], "E": [5., 5]}, index=idx)
    view = AsOfView({"close": cl})
    res = backtest(CrossSectionalStrategy(factor, quantile=0.2), view, costs_bps=0.0)
    # t0: ロングE(+10%) − ショートA(−5%) = 0.10 − (−0.05) = 0.15
    assert res.gross.loc[idx[0]] == pytest.approx(0.15)


def test_costs_reduce_returns():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    op = pd.DataFrame({"A": [100., 100, 88, 99]}, index=idx)
    cl = pd.DataFrame({"A": [100., 100, 90, 99]}, index=idx)
    view = AsOfView({"open": op, "close": cl})
    free = backtest(GapReversal(0.10), view, costs_bps=0.0).returns.loc[idx[2]]
    costed = backtest(GapReversal(0.10), view, costs_bps=50.0).returns.loc[idx[2]]
    assert costed < free


def test_execution_lag_shifts_realization():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.DataFrame({"A": [100., 110, 99, 108.9]}, index=idx)  # +10/-10/+10%
    sig = pd.Series(1.0, index=idx)                 # 常時ロング
    strat = SignalTimingStrategy(sig, "A", threshold=0.0)
    view = AsOfView({"close": close})
    r0 = backtest(strat, view, costs_bps=0.0, execution_lag=0).returns
    assert r0.loc[idx[0]] == pytest.approx(0.10)    # 決定足終値で執行: t0→t1
    r1 = backtest(strat, view, costs_bps=0.0, execution_lag=1).returns
    assert r1.loc[idx[0]] == pytest.approx(-0.10)   # 翌足で執行: t1→t2
    assert idx[2] not in r1.index                   # 末尾2本は実現不可で除外


def _xs_setup():
    # CrossSectionalStrategy は最低5銘柄必要。6銘柄, q=0.2 → long F / short A。
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    cols = ["A", "B", "C", "D", "E", "F"]
    close = pd.DataFrame({c: [100., 101] for c in cols}, index=idx)
    factor = pd.DataFrame({c: [i + 1.0, i + 1.0] for i, c in enumerate(cols)},
                          index=idx)
    adv = pd.DataFrame({c: [2e8, 2e8] for c in cols}, index=idx)
    adv["A"] = [1e8, 1e8]                            # A（ショート側）が最も薄い
    return idx, close, factor, adv


def test_capacity_from_adv():
    idx, close, factor, adv = _xs_setup()
    res = backtest(CrossSectionalStrategy(factor, quantile=0.2),
                   AsOfView({"close": close}), costs_bps=0.0, adv=adv,
                   participation=0.1)
    # long F / short A, |w|=1ずつ → 容量 = min(0.1×1e8/1, 0.1×2e8/1) = 1e7（A律速）
    assert res.capacity_jpy == pytest.approx(1e7)


def test_capacity_nan_without_adv():
    idx, close, factor, adv = _xs_setup()
    res = backtest(CrossSectionalStrategy(factor, quantile=0.2),
                   AsOfView({"close": close}), costs_bps=0.0)
    assert np.isnan(res.capacity_jpy)


def test_rebalance_must_be_contiguous_panel_subsequence():
    # 疎な rebalance は「次の1バー分」しか実現せず中間リターンが脱落＝拒否する。
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    close = pd.DataFrame({"A": np.linspace(100.0, 109.0, 10)}, index=idx)
    strat = SignalTimingStrategy(pd.Series(1.0, index=idx), "A", threshold=0.0)
    view = AsOfView({"close": close})
    r = backtest(strat, view, costs_bps=0.0, rebalance=idx[3:8]).returns
    assert list(r.index) == list(idx[3:8])      # 連続部分列（ウォームアップ）は可
    with pytest.raises(ValueError, match="疎"):
        backtest(strat, view, costs_bps=0.0, rebalance=idx[::2])
    with pytest.raises(ValueError, match="無い日付"):
        backtest(strat, view, costs_bps=0.0,
                 rebalance=pd.DatetimeIndex(["2030-01-01", "2030-01-02"]))


def test_ann_factor_inference():
    daily = pd.date_range("2020-01-01", periods=252 * 3, freq="B")
    assert 230 < _ann_factor(daily) < 270          # 日次 ≈ 252
    monthly = pd.date_range("2016-01-31", periods=120, freq="ME")
    assert 11.5 < _ann_factor(monthly) < 12.5       # 月次 ≈ 12


# --- 執行現実性（値幅制限の執行不能・貸株コスト）-----------------------------

def _timing_setup():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.DataFrame({"A": [100., 110, 121, 133.1]}, index=idx)  # 毎日+10%
    flags = pd.DataFrame(False, index=idx, columns=["A"])
    return idx, close, flags


def test_no_buy_blocks_entry_then_enters():
    idx, close, flags = _timing_setup()
    sig = pd.Series(1.0, index=idx)                       # 常時ロング希望
    strat = SignalTimingStrategy(sig, "A", threshold=0.0)
    nb = flags.copy()
    nb.loc[idx[0], "A"] = True                            # d0 はストップ高引け＝買えない
    view = AsOfView({"close": close})
    res = backtest(strat, view, costs_bps=0.0, no_buy=nb)
    assert res.returns.loc[idx[0]] == 0.0                 # 建てられず現金
    assert res.n_blocked.loc[idx[0]] == 1
    assert res.n_positions.loc[idx[0]] == 0
    assert res.returns.loc[idx[1]] == pytest.approx(0.10)  # 翌日に建つ
    assert res.turnover.loc[idx[1]] == pytest.approx(1.0)


def test_no_sell_blocks_exit_and_carries_position():
    idx, close, flags = _timing_setup()
    sig = pd.Series([1.0, 0.0, 0.0, 0.0], index=idx)      # d1 で手仕舞いたい
    strat = SignalTimingStrategy(sig, "A", threshold=0.5)
    ns = flags.copy()
    ns.loc[idx[1], "A"] = True                            # d1 はストップ安引け＝売れない
    view = AsOfView({"close": close})
    res = backtest(strat, view, costs_bps=0.0, no_sell=ns)
    assert res.returns.loc[idx[1]] == pytest.approx(0.10)  # 持ち越し＝d1→d2 を被る
    assert res.n_blocked.loc[idx[1]] == 1
    assert res.n_positions.loc[idx[1]] == 1
    assert res.n_positions.loc[idx[2]] == 0               # 翌日に決済できた
    assert res.turnover.loc[idx[2]] == pytest.approx(1.0)


def test_all_false_flags_match_baseline():
    idx, close, factor, adv = _xs_setup()
    view = AsOfView({"close": close})
    base = backtest(CrossSectionalStrategy(factor, quantile=0.2), view,
                    costs_bps=15.0)
    nb = pd.DataFrame(False, index=close.index, columns=close.columns)
    flagged = backtest(CrossSectionalStrategy(factor, quantile=0.2), view,
                       costs_bps=15.0, no_buy=nb, no_sell=nb.copy())
    pd.testing.assert_series_equal(base.returns, flagged.returns)
    pd.testing.assert_series_equal(base.turnover, flagged.turnover)


def test_no_buy_respects_execution_lag():
    idx, close, flags = _timing_setup()
    sig = pd.Series(1.0, index=idx)
    strat = SignalTimingStrategy(sig, "A", threshold=0.0)
    nb = flags.copy()
    nb.loc[idx[1], "A"] = True                  # 執行バー（t0+lag1=d1）が張り付き
    view = AsOfView({"close": close})
    res = backtest(strat, view, costs_bps=0.0, execution_lag=1, no_buy=nb)
    assert res.returns.loc[idx[0]] == 0.0       # d1 執行できず
    assert res.n_blocked.loc[idx[0]] == 1


def test_short_borrow_cost_charged_on_short_gross():
    idx, close, factor, adv = _xs_setup()
    view = AsOfView({"close": close})
    strat = CrossSectionalStrategy(factor, quantile=0.2)   # long F / short A（|短|=1）
    free = backtest(strat, view, costs_bps=0.0)
    costed = backtest(strat, view, costs_bps=0.0, short_borrow_bps=120.0)
    assert costed.short_gross.loc[idx[0]] == pytest.approx(1.0)
    per_period = 120.0 / 1e4 / _ann_factor(pd.DatetimeIndex(free.returns.index))
    diff = free.returns.loc[idx[0]] - costed.returns.loc[idx[0]]
    assert diff == pytest.approx(per_period)               # 短グロス×期間按分の控除
    pd.testing.assert_series_equal(free.gross, costed.gross)  # gross は不変


# --- 状態依存コスト（銘柄×日付の bps パネル）---------------------------------

def test_cost_panel_charges_per_name():
    idx, close, factor, adv = _xs_setup()        # long F / short A（|Δw|=1 ずつ）
    view = AsOfView({"close": close})
    panel = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    panel["A"] = 100.0                           # A だけ 100bps、他は 0bps
    base = backtest(CrossSectionalStrategy(factor, 0.2), view, costs_bps=0.0)
    var = backtest(CrossSectionalStrategy(factor, 0.2), view, costs_bps=panel)
    diff = base.returns.loc[idx[0]] - var.returns.loc[idx[0]]
    assert diff == pytest.approx(1.0 * 100.0 / 1e4)        # |Δw_A|×100bp のみ課金


def test_cost_panel_constant_equals_scalar():
    idx, close, factor, adv = _xs_setup()
    view = AsOfView({"close": close})
    panel = pd.DataFrame(15.0, index=close.index, columns=close.columns)
    s = backtest(CrossSectionalStrategy(factor, 0.2), view, costs_bps=15.0)
    p = backtest(CrossSectionalStrategy(factor, 0.2), view, costs_bps=panel)
    pd.testing.assert_series_equal(s.returns, p.returns)


# --- リバランス・デッドバンド（rebalance_band・docs/04 P2-A）------------------

class _SeqStrategy:
    """日付→ウェイトの逐次表を返すテスト用戦略。"""

    name, params = "seq", {}

    def __init__(self, table):
        self._t = table

    def target_weights(self, asof):
        return self._t.get(asof.asof, pd.Series(dtype="float64"))


def _band_view():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.DataFrame({"A": [100.0] * 4, "B": [100.0] * 4}, index=idx)
    return idx, AsOfView({"close": close})


def test_band_zero_matches_baseline():
    idx, view = _band_view()
    table = {idx[0]: pd.Series({"A": 0.5}), idx[1]: pd.Series({"A": 0.502}),
             idx[2]: pd.Series({"A": 0.6})}
    base = backtest(_SeqStrategy(table), view, costs_bps=10.0)
    banded = backtest(_SeqStrategy(table), view, costs_bps=10.0,
                      rebalance_band=0.0)
    pd.testing.assert_series_equal(base.returns, banded.returns)
    pd.testing.assert_series_equal(base.turnover, banded.turnover)


def test_band_suppresses_small_orders_and_dust_exits():
    idx, view = _band_view()
    table = {idx[0]: pd.Series({"A": 0.5}),          # 初回建玉（band 超）
             idx[1]: pd.Series({"A": 0.502}),        # |Δ|=0.002 < band → 据え置き
             idx[2]: pd.Series({"A": 0.503})}        # 0.5 基準で |Δ|=0.003 < band
    res = backtest(_SeqStrategy(table), view, costs_bps=10.0,
                   rebalance_band=0.005)
    assert res.turnover.loc[idx[0]] == pytest.approx(0.5)
    assert res.turnover.loc[idx[1]] == pytest.approx(0.0)   # 微調整は取引しない
    assert res.turnover.loc[idx[2]] == pytest.approx(0.0)   # 基準は保有（キャリー）
    # ダスト清算の抑制：目標 0 でも |0 − 0.004| < band なら保有を維持
    table2 = {idx[0]: pd.Series({"A": 0.004}), idx[1]: pd.Series(dtype="float64")}
    res2 = backtest(_SeqStrategy(table2), view, costs_bps=10.0,
                    rebalance_band=0.005)
    assert res2.turnover.loc[idx[0]] == pytest.approx(0.0)  # 建玉自体が band 未満
    assert res2.n_positions.loc[idx[1]] == 0


def test_band_trades_when_delta_exceeds_threshold():
    idx, view = _band_view()
    table = {idx[0]: pd.Series({"A": 0.5}), idx[1]: pd.Series({"A": 0.6})}
    res = backtest(_SeqStrategy(table), view, costs_bps=10.0,
                   rebalance_band=0.005)
    assert res.turnover.loc[idx[1]] == pytest.approx(0.1)   # band 超は全量執行


def test_apply_rebalance_band_pure_function_matches_engine_semantics():
    idx, view = _band_view()
    table = {idx[0]: pd.Series({"A": 0.5, "B": -0.5}),
             idx[1]: pd.Series({"A": 0.502, "B": -0.51}),
             idx[2]: pd.Series({"B": -0.5})}
    banded = apply_rebalance_band(table, 0.005)
    assert banded[idx[1]]["A"] == pytest.approx(0.5)        # 据え置き
    assert banded[idx[1]]["B"] == pytest.approx(-0.51)      # band 超は更新
    assert "A" not in banded[idx[2]].index                  # |0−0.5|≥band → 清算
    assert banded[idx[2]]["B"] == pytest.approx(-0.5)
    ident = apply_rebalance_band(table, 0.0)                # band=0 は恒等
    assert ident[idx[0]] is table[idx[0]]


def test_band_applies_after_no_buy_carry():
    # キャリー後ウェイト基準：ブロックで据え置かれた銘柄は band 判定でも据え置き。
    idx, view = _band_view()
    table = {idx[0]: pd.Series({"A": 0.5}), idx[1]: pd.Series({"A": 0.502})}
    nb = pd.DataFrame(True, index=idx, columns=["A"])
    res = backtest(_SeqStrategy(table), view, costs_bps=10.0,
                   no_buy=nb, rebalance_band=0.005)
    assert res.turnover.sum() == pytest.approx(0.0)          # 一度も建たない
    assert int(res.n_blocked.loc[idx[0]]) == 1


# --- T+1 始値執行リプレイ（open_fill_backtest・DP17）--------------------------

def _daily_open():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    open_px = pd.DataFrame({"A": np.linspace(100.0, 109.0, 10)}, index=idx)
    return idx, open_px


def test_open_fill_executes_next_day_open():
    idx, open_px = _daily_open()
    w = {idx[1]: pd.Series({"A": 1.0}),          # 決定 d1 → 約定 d2
         idx[4]: pd.Series({"A": 1.0}),          # 決定 d4 → 約定 d5
         idx[7]: pd.Series({"A": 1.0})}          # 最後の決定は評価不能で脱落
    res = open_fill_backtest(w, open_px, costs_bps=10.0)
    assert list(res.returns.index) == [idx[1], idx[4]]
    exp1 = open_px.loc[idx[5], "A"] / open_px.loc[idx[2], "A"] - 1.0
    assert res.gross.loc[idx[1]] == pytest.approx(exp1)    # open(d5)/open(d2)-1
    assert res.returns.loc[idx[1]] == pytest.approx(exp1 - 10.0 / 1e4 * 1.0)
    assert res.turnover.loc[idx[4]] == pytest.approx(0.0)  # 同一ウェイト→無回転
    assert res.returns.loc[idx[4]] == pytest.approx(
        open_px.loc[idx[8], "A"] / open_px.loc[idx[5], "A"] - 1.0)


def test_open_fill_decision_close_not_used():
    # 決定日の終値が約定に使われないこと＝寄りギャップを負担することの確認。
    idx, open_px = _daily_open()
    open_px.loc[idx[2], "A"] = 200.0             # 約定日の寄りが大きく窓開け
    w = {idx[1]: pd.Series({"A": 1.0}), idx[4]: pd.Series({"A": 1.0})}
    res = open_fill_backtest(w, open_px, costs_bps=0.0)
    exp = open_px.loc[idx[5], "A"] / 200.0 - 1.0           # 不利な寄りで建つ
    assert res.gross.loc[idx[1]] == pytest.approx(exp)


def test_open_fill_borrow_and_cost_panel():
    idx, open_px = _daily_open()
    w = {idx[1]: pd.Series({"A": -1.0}), idx[4]: pd.Series({"A": -1.0})}
    panel = pd.DataFrame(50.0, index=idx, columns=["A"])   # 約定日の行を参照
    res = open_fill_backtest(w, open_px, costs_bps=panel, short_borrow_bps=120.0)
    ann = _ann_factor(pd.DatetimeIndex([idx[1]]))          # 決定日列から推定（<3→252）
    exp_gross = open_px.loc[idx[5], "A"] / open_px.loc[idx[2], "A"] - 1.0
    # gross = w·rel = -1×(+ret)。net = gross − cost(1×50bp) − borrow(120bp/ann×1)
    assert res.gross.loc[idx[1]] == pytest.approx(-exp_gross)
    assert res.returns.loc[idx[1]] == pytest.approx(
        -exp_gross - 50.0 / 1e4 - 120.0 / 1e4 / ann)
    assert res.short_gross.loc[idx[1]] == pytest.approx(1.0)


def test_open_fill_nan_open_skips_pnl():
    idx, open_px = _daily_open()
    open_px.loc[idx[2], "A"] = np.nan            # 約定日の寄りが無い（売買不成立）
    w = {idx[1]: pd.Series({"A": 1.0}), idx[4]: pd.Series({"A": 1.0})}
    res = open_fill_backtest(w, open_px, costs_bps=0.0)
    assert res.gross.loc[idx[1]] == pytest.approx(0.0)     # 損益に寄与しない
