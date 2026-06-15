"""C1（TOB リスクアーブ）診断ロジックの検証。ネットワーク不要。"""
import pandas as pd

from invest_system.equities import tob_arb as ta


def test_subperiod_regime_split():
    assert ta.subperiod(2015) is None        # 検証窓外
    assert ta.subperiod(2016) == "2016-2019"
    assert ta.subperiod(2019) == "2016-2019"
    assert ta.subperiod(2020) == "2020-2026"
    assert ta.subperiod(2026) == "2020-2026"


def test_mark_competing_same_target_near():
    d = pd.DataFrame({
        "target_code": ["1234", "1234", "5678"],
        "announce_date": ["2019-01-10", "2019-03-01", "2020-05-01"],
    })
    comp = ta.mark_competing(d, window_days=180)
    assert comp.tolist() == [True, True, False]      # 1234 が近接競合、5678 は単独


def test_mark_competing_same_target_far_apart():
    # 同一対象でも 180 日超離れていれば競合でない（別案件）
    d = pd.DataFrame({
        "target_code": ["1234", "1234"],
        "announce_date": ["2018-01-01", "2020-01-01"],
    })
    assert ta.mark_competing(d, window_days=180).tolist() == [False, False]


def test_deal_metrics_success_uses_final_price():
    # エントリー信号は当初価格/始値、成立リターンは最終価格（バンプ後）で出口
    m = ta.deal_metrics("成立", initial_price=1000, final_price=1100,
                        prev_close=800, entry_open=1000, exit_mkt=None)
    assert abs(m["premium"] - (1000 / 800 - 1)) < 1e-9
    assert abs(m["arb_spread"] - 0.0) < 1e-9
    assert abs(m["ret"] - (1100 / 1000 - 1)) < 1e-9      # 最終価格で受取


def test_deal_metrics_failure_uses_market_exit():
    m = ta.deal_metrics("不成立", initial_price=1000, final_price=1000,
                        prev_close=900, entry_open=950, exit_mkt=850)
    assert abs(m["arb_spread"] - (1000 / 950 - 1)) < 1e-9
    assert abs(m["ret"] - (850 / 950 - 1)) < 1e-9        # 発表前水準への急落＝損

def test_deal_metrics_missing_entry_returns_premium_only():
    m = ta.deal_metrics("成立", initial_price=1000, final_price=1000,
                        prev_close=800, entry_open=None, exit_mkt=None)
    assert m["arb_spread"] is None and m["ret"] is None
    assert abs(m["premium"] - (1000 / 800 - 1)) < 1e-9


def test_live_weights_equal_weight_with_cash_drag():
    sched = pd.DataFrame({
        "sec": ["13010", "13050", "13060"],
        "entry_date": pd.to_datetime(["2022-01-05", "2022-01-20", "2022-03-01"]),
        "exit_date": pd.to_datetime(["2022-02-15", "2022-02-28", "2022-04-10"]),
    })
    # 2022-01-25：13010 と 13050 がライブ（13060 未開始）→ 各 0.5
    w = ta.live_weights(sched, pd.Timestamp("2022-01-25"))
    assert set(w.index) == {"13010", "13050"}
    assert all(abs(v - 0.5) < 1e-9 for v in w.values)
    # 2022-03-05：13060 のみライブ → 1.0（他は exit 済）
    w2 = ta.live_weights(sched, pd.Timestamp("2022-03-05"))
    assert w2.index.tolist() == ["13060"] and abs(w2.iloc[0] - 1.0) < 1e-9
    # 2022-05-01：全件 exit 済 → 空（現金 100%）
    assert ta.live_weights(sched, pd.Timestamp("2022-05-01")).empty


def test_spread_bucket_edges():
    assert ta.spread_bucket(0.005) == "<1%"
    assert ta.spread_bucket(0.01) == "1-3%"
    assert ta.spread_bucket(0.03) == "3-7%"
    assert ta.spread_bucket(0.07) == ">=7%"
    assert ta.spread_bucket(None) is None
