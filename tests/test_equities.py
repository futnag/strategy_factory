"""日本株クロスセクション基盤（純関数）の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.universe import (
    apply_universe_mask,
    filter_common_stocks,
    liquid_universe_mask,
    point_in_time_universe,
    select_universe,
    universe_members,
)
from invest_system.equities.panel import (
    assemble_panel,
    delisting_mask,
    forward_returns,
    impute_delisting,
    load_daily_panel,
    total_forward_returns,
    trailing_momentum,
)
from invest_system.equities.fundamentals import (
    fundamentals_panel, load_fundamentals, point_in_time,
)
from invest_system.equities.factors import (
    cross_sectional_residualize,
    cross_sectional_zscore,
    low_volatility,
    market_cap,
    residual_momentum,
    sector_neutralize,
    value_quality_size_factors,
)
from invest_system.equities.backtest import long_short_returns


# --- universe ---------------------------------------------------------------
def test_filter_common_stocks_excludes_etf():
    listed = pd.DataFrame({
        "Code": ["13010", "13060", "72030", "99840"],
        "Mkt": ["0111", "0109", "0111", "0112"],  # 1306 は ETF(0109)
    })
    out = filter_common_stocks(listed)
    assert set(out["Code"]) == {"13010", "72030", "99840"}


def test_select_universe_by_liquidity():
    listed = pd.DataFrame({"Code": ["A", "B", "C"], "Mkt": ["0111"] * 3})
    # B が最も売買代金大、C は観測不足
    turnover = pd.DataFrame({
        "A": [1.0, 1.0, 1.0],
        "B": [9.0, 9.0, 9.0],
        "C": [5.0, np.nan, np.nan],
    }, index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    uni = select_universe(listed, turnover, top_n=2, min_obs=2)
    assert uni == ["B", "A"]  # 流動性順、C は min_obs 未満で除外


def test_point_in_time_universe_no_lookahead():
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    turn = pd.DataFrame({"A": [10., 10, 10], "B": [20., 20, 20],
                         "C": [30., 30, 30], "D": [1., 100, 500]}, index=idx)
    mask = point_in_time_universe(turn, top_n=2, lookback=1, min_obs=1)
    # t0: 当時 D=1 は下位 → 未来の急増(t1,t2)を使わない＝先読み無し
    assert bool(mask.loc[idx[0], "C"]) and bool(mask.loc[idx[0], "B"])
    assert not bool(mask.loc[idx[0], "D"])
    # t1/t2: D が上位に入る
    assert bool(mask.loc[idx[1], "D"]) and bool(mask.loc[idx[2], "D"])
    assert universe_members(mask) == ["B", "C", "D"]   # A は一度も入らない


def test_apply_universe_mask():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    fac = pd.DataFrame({"A": [1., 2], "B": [3., 4], "C": [5., 6]}, index=idx)
    mask = pd.DataFrame({"A": [True, False], "B": [True, True], "C": [False, True]},
                        index=idx)
    out = apply_universe_mask(fac, mask)
    assert out.loc[idx[0], "A"] == 1.0 and np.isnan(out.loc[idx[0], "C"])
    assert np.isnan(out.loc[idx[1], "A"]) and out.loc[idx[1], "C"] == 6.0


# --- panel ------------------------------------------------------------------
def test_assemble_panel_and_dedup():
    snaps = {
        pd.Timestamp("2024-01-31"): pd.DataFrame(
            {"Code": ["A", "B", "A"], "AdjC": [100.0, 200.0, 101.0]}),
        pd.Timestamp("2024-02-29"): pd.DataFrame(
            {"Code": ["A", "B"], "AdjC": [110.0, 190.0]}),
    }
    panel = assemble_panel(snaps, "AdjC")
    assert list(panel.columns) == ["A", "B"]
    assert panel.loc[pd.Timestamp("2024-01-31"), "A"] == 101.0  # 重複は last
    assert panel.loc[pd.Timestamp("2024-02-29"), "B"] == 190.0


def test_forward_returns_alignment_no_lookahead():
    price = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 99.0]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    fr = forward_returns(price)
    assert fr.iloc[0]["A"] == pytest.approx(0.10)   # t0→t1
    assert fr.iloc[0]["B"] == pytest.approx(-0.10)
    assert fr.iloc[1]["A"] == pytest.approx(0.10)
    assert np.isnan(fr.iloc[2]["A"])                # 最終行は未実現


def test_trailing_momentum_skip():
    price = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1]},
        index=pd.date_range("2024-01-31", periods=4, freq="ME"))
    mom = trailing_momentum(price, lookback=2, skip=1)
    assert mom.iloc[2]["A"] == pytest.approx(0.10)  # 110/100-1
    assert mom.iloc[3]["A"] == pytest.approx(0.10)  # 121/110-1
    assert np.isnan(mom.iloc[1]["A"])


# --- fundamentals: ポイントインタイム（先読み排除）-------------------------
def test_point_in_time_no_lookahead_and_asof():
    fund = pd.DataFrame({
        "Code": ["100", "100"],
        "DiscDate": pd.to_datetime(["2024-01-15", "2024-04-20"]),
        "EPS": [10.0, 20.0],
    })
    rebal = pd.to_datetime(["2024-01-10", "2024-01-31", "2024-04-19", "2024-04-30"])
    pit = point_in_time(fund, rebal, ["EPS"], lag_days=1)
    eps = pit["EPS"]["100"]
    assert np.isnan(eps.loc["2024-01-10"])             # 開示前 → NaN
    assert eps.loc["2024-01-31"] == 10.0               # 1月開示が反映
    assert eps.loc["2024-04-19"] == 10.0               # 4/20開示は 4/19 時点で未反映
    assert eps.loc["2024-04-30"] == 20.0               # 4月開示が反映


def test_point_in_time_same_day_excluded_by_lag():
    fund = pd.DataFrame({
        "Code": ["100"], "DiscDate": pd.to_datetime(["2024-01-15"]), "EPS": [10.0]})
    rebal = pd.to_datetime(["2024-01-15"])            # 開示当日
    pit = point_in_time(fund, rebal, ["EPS"], lag_days=1)
    assert np.isnan(pit["EPS"]["100"].loc["2024-01-15"])  # lag=1 で当日は使わない


def test_load_fundamentals_unions_and_dedupes(tmp_path):
    bd = tmp_path / "fins_summary"; bd.mkdir()
    bc = tmp_path / "statements"; bc.mkdir()
    # by-date ミラー：ある開示日の全社（2社）
    pd.DataFrame({"DiscDate": ["2024-05-10", "2024-05-10"], "Code": ["7203", "6758"],
                  "DiscNo": ["1", "2"], "EPS": [10.0, 20.0]}
                 ).to_parquet(bd / "20240510.parquet")
    # 旧 by-code：7203 の履歴（DiscNo=1 は by-date と重複、DiscNo=0 は追加の古い開示）
    pd.DataFrame({"DiscDate": ["2024-05-10", "2024-02-10"], "Code": ["7203", "7203"],
                  "DiscNo": ["1", "0"], "EPS": [10.0, 8.0]}
                 ).to_parquet(bc / "7203.parquet")
    pd.DataFrame({"_empty": pd.Series([], dtype="bool")}  # 空マーカーは無視
                 ).to_parquet(bd / "20240101.parquet")
    out = load_fundamentals(base=str(tmp_path))
    assert len(out) == 3                                   # 7203:2(重複1除去)＋6758:1
    only = load_fundamentals(codes=["7203"], base=str(tmp_path))
    assert set(only["Code"]) == {"7203"} and len(only) == 2
    # 1呼び出しで as-of パネル組立（5/10開示は lag=1 で 5/13 に反映、5/10当日は未反映）
    panel = fundamentals_panel(pd.to_datetime(["2024-05-10", "2024-05-13"]),
                               ["EPS"], base=str(tmp_path), lag_days=1)
    assert np.isnan(panel["EPS"]["6758"].loc["2024-05-10"])
    assert panel["EPS"]["6758"].loc["2024-05-13"] == 20.0


def test_load_daily_panel_wide_and_skips_empty(tmp_path):
    dd = tmp_path / "daily"; dd.mkdir()
    pd.DataFrame({"Date": ["2024-05-10", "2024-05-10"], "Code": ["7203", "6758"],
                  "AdjC": [100.0, 200.0], "Va": [1e9, 2e9]}
                 ).to_parquet(dd / "20240510.parquet")
    pd.DataFrame({"Date": ["2024-05-13", "2024-05-13"], "Code": ["7203", "6758"],
                  "AdjC": [110.0, 190.0], "Va": [1.1e9, 1.9e9]}
                 ).to_parquet(dd / "20240513.parquet")
    pd.DataFrame({"_empty": pd.Series([], dtype="bool")}    # 祝日マーカーは無視
                 ).to_parquet(dd / "20240511.parquet")
    panel = load_daily_panel(field="AdjC", base=str(tmp_path))
    assert list(panel.index) == [pd.Timestamp("2024-05-10"), pd.Timestamp("2024-05-13")]
    assert sorted(panel.columns) == ["6758", "7203"]
    assert panel.loc["2024-05-13", "7203"] == 110.0
    va = load_daily_panel(field="Va", codes=["7203"], base=str(tmp_path))
    assert list(va.columns) == ["7203"] and va.loc["2024-05-10", "7203"] == 1e9


def test_load_daily_panel_silver_fastpath(tmp_path, monkeypatch):
    """base=None の既定パスが Silver(processed/equities/wide) を別名解決で読む（回帰）。"""
    from invest_system.data.sources import jquants as jq
    wd = tmp_path / "processed" / "equities" / "wide"; wd.mkdir(parents=True)
    pd.DataFrame({"7203": [100.0, 110.0], "6758": [200.0, 190.0]},
                 index=pd.DatetimeIndex(["2024-05-10", "2024-05-13"], name="Date")
                 ).to_parquet(wd / "adj_close.parquet")
    monkeypatch.setattr(jq, "_CACHE", tmp_path / "jquants")     # parent=tmp を data root に
    panel = load_daily_panel(field="AdjC")                      # base=None＝Silver 高速パス
    assert sorted(panel.columns) == ["6758", "7203"]
    assert panel.loc["2024-05-13", "7203"] == 110.0
    sub = load_daily_panel(field="AdjC", codes=["7203"])       # codes 絞り込み
    assert list(sub.columns) == ["7203"]


# --- factors ----------------------------------------------------------------
def test_market_cap():
    price = pd.DataFrame({"A": [10.0]}, index=[pd.Timestamp("2024-01-31")])
    shares = pd.DataFrame({"A": [1000.0]}, index=[pd.Timestamp("2024-01-31")])
    tr = pd.DataFrame({"A": [100.0]}, index=[pd.Timestamp("2024-01-31")])
    mc = market_cap(price, shares, tr)
    assert mc.loc[pd.Timestamp("2024-01-31"), "A"] == 9000.0  # 10*(1000-100)


def test_cross_sectional_zscore():
    df = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]},
                      index=[pd.Timestamp("2024-01-31")])
    z = cross_sectional_zscore(df, winsor=10)
    row = z.iloc[0]
    assert row["B"] == pytest.approx(0.0)
    assert row["A"] == pytest.approx(-1.224744871, rel=1e-6)
    assert row["C"] == pytest.approx(1.224744871, rel=1e-6)


def test_sector_neutralize_within_group_demean():
    df = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [10.0], "D": [20.0]},
                      index=[pd.Timestamp("2024-01-31")])
    sector = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})
    out = sector_neutralize(df, sector)
    r = out.iloc[0]
    assert r["A"] == pytest.approx(-1.0) and r["B"] == pytest.approx(1.0)
    assert r["C"] == pytest.approx(-5.0) and r["D"] == pytest.approx(5.0)


# --- backtest ---------------------------------------------------------------
def test_residualize_removes_explainable_part():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    cols = ["A", "B", "C", "D", "E"]
    control = pd.DataFrame([[1., 2, 3, 4, 5], [2., 4, 6, 8, 10]], index=idx, columns=cols)
    target = 2.0 * control + 5.0          # control で完全に説明可能 → 残差≈0
    resid = cross_sectional_residualize(target, [control])
    assert np.allclose(resid.loc[idx[0]].to_numpy(), 0.0, atol=1e-7)


def test_residualize_keeps_independent_part():
    idx = pd.date_range("2024-01-31", periods=1, freq="ME")
    cols = ["A", "B", "C", "D", "E", "F"]
    control = pd.DataFrame([[1., 2, 3, 4, 5, 6]], index=idx, columns=cols)
    indep = np.array([1., -1, 1, -1, 1, -1])
    target = pd.DataFrame([control.iloc[0].to_numpy() + indep], index=idx, columns=cols)
    resid = cross_sectional_residualize(target, [control]).loc[idx[0]].to_numpy()
    assert resid.std() > 0                                  # 独立成分は残る
    assert abs(np.corrcoef(resid, control.iloc[0].to_numpy())[0, 1]) < 1e-6  # 直交


def test_low_volatility_sign_and_no_lookahead():
    idx = pd.date_range("2024-01-31", periods=14, freq="ME")
    smooth = pd.Series(100.0 * (1.01 ** np.arange(14)), index=idx)          # 低ボラ
    zigzag = pd.Series(100.0 * (1 + 0.1 * ((-1.0) ** np.arange(14))), index=idx)  # 高ボラ
    lv = low_volatility(pd.DataFrame({"A": smooth, "B": zigzag}), window=6)
    assert lv.iloc[:2].isna().all().all()                  # 過去不足は NaN（先読みなし）
    assert lv.iloc[-1]["A"] > lv.iloc[-1]["B"]             # 低ボラ A が大（ロング側）


def test_accruals_quality_sign_in_bundle():
    idx = pd.to_datetime(["2024-03-31"])

    def one(a, b):
        return pd.DataFrame({"100": [a], "200": [b]}, index=idx)

    raw = one(1000.0, 1000.0)
    pit = {"CFO": one(50.0, 10.0), "NP": one(10.0, 50.0), "TA": one(100.0, 100.0),
           "ShOutFY": one(1.0, 1.0), "TrShFY": one(0.0, 0.0), "Eq": one(1.0, 1.0)}
    out = value_quality_size_factors(pit, raw)
    # accruals = (CFO − NP)/TA：100→(50−10)/100=+0.4（高品質）, 200→−0.4（低品質）
    assert out["accruals"].loc[idx[0], "100"] == pytest.approx(0.4)
    assert out["accruals"].loc[idx[0], "200"] == pytest.approx(-0.4)


def test_residual_momentum_extracts_idiosyncratic_component():
    # 市場ベータ起因の成分は残差化で消え、直近窓の固有ドリフトだけが符号に出る。
    rng = np.random.default_rng(7)
    n = 40
    idx = pd.date_range("2018-01-31", periods=n, freq="ME")
    mkt = pd.Series(rng.normal(0.0, 0.04, n), index=idx)
    noise = rng.normal(0.0, 0.001, (n, 3))
    ret = pd.DataFrame({"A": mkt + noise[:, 0], "B": mkt + noise[:, 1],
                        "C": 2.0 * mkt + noise[:, 2]}, index=idx)
    # 最終行 t の momentum 窓＝月 t-12..t-2（iloc -13..-3）に固有ショックを置く
    ret.iloc[-13:-2, ret.columns.get_loc("A")] += 0.02
    ret.iloc[-13:-2, ret.columns.get_loc("B")] -= 0.02
    rm = residual_momentum(ret, mkt, window=36, mom_len=12, skip=1)
    last = rm.iloc[-1]
    assert last["A"] > 1.0 and last["B"] < -1.0     # 固有成分は強く符号に出る
    assert abs(last["C"]) < abs(last["A"]) / 5      # 高ベータだけでは出ない
    assert np.isnan(rm.iloc[35]).all() and not np.isnan(rm.iloc[36]["A"])


def test_residual_momentum_pit_and_window_rule():
    rng = np.random.default_rng(8)
    n = 45
    idx = pd.date_range("2018-01-31", periods=n, freq="ME")
    mkt = pd.Series(rng.normal(0.0, 0.03, n), index=idx)
    ret = pd.DataFrame({"A": mkt + rng.normal(0, 0.01, n),
                        "D": mkt + rng.normal(0, 0.01, n)}, index=idx)
    ret.iloc[5, ret.columns.get_loc("D")] = np.nan   # 窓に欠損 → 該当行は NaN
    rm1 = residual_momentum(ret, mkt)
    assert np.isnan(rm1.iloc[38]["D"])               # 36ヶ月窓に欠損月を含む
    assert not np.isnan(rm1.iloc[42]["D"])           # 欠損が窓から抜ければ復帰
    ret2 = ret.copy()
    ret2.iloc[-1] = 9.9                              # 未来（当月リターン）を改変
    rm2 = residual_momentum(ret2, mkt)
    assert rm2.iloc[-1]["A"] == rm1.iloc[-1]["A"]    # 行 t は ≤t-1 のみ参照＝PIT


def test_negative_shares_yield_nan_not_sign_flip():
    # 片側欠損（ShOutFY 無し×TrShFY 有り）で株数が負→時価総額の符号反転を防ぐ。
    idx = [pd.Timestamp("2024-01-31")]

    def one(a, b):
        return pd.DataFrame({"100": [a], "200": [b]}, index=idx)

    raw = one(1000.0, 1000.0)
    mc = market_cap(raw, one(np.nan, 10.0), treasury=one(2.0, 1.0))
    assert np.isnan(mc.loc[idx[0], "100"])
    assert mc.loc[idx[0], "200"] == pytest.approx(9000.0)
    pit = {"ShOutFY": one(np.nan, 10.0), "TrShFY": one(2.0, 1.0),
           "Eq": one(5.0, 5.0)}
    out = value_quality_size_factors(pit, raw)
    assert np.isnan(out["book_to_market"].loc[idx[0], "100"])  # 負のB/Mに化けない
    assert out["book_to_market"].loc[idx[0], "200"] > 0


def test_long_short_returns_no_lookahead_on_delisting():
    # 「翌期リターンが存在するか」を選定に使わない（生存者バイアス排除）。
    # E はファクター最上位だが翌期に退出（fwd NaN）→ ロングに選ばれ寄与は 0。
    idx = [pd.Timestamp("2024-01-31")]
    factor = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0],
                           "E": [5.0]}, index=idx)
    fwd = pd.DataFrame({"A": [-0.05], "B": [0.0], "C": [0.0], "D": [0.10],
                        "E": [np.nan]}, index=idx)
    r = long_short_returns(factor, fwd, quantile=0.2, min_names=4)
    # long=E（D=+10% に化けない）, short=A（+0.05）
    assert r.iloc[0] == pytest.approx(0.05)


def test_long_short_returns_sign_and_costs():
    idx = [pd.Timestamp("2024-01-31")]
    factor = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]}, index=idx)
    fwd = pd.DataFrame({"A": [-0.05], "B": [0.0], "C": [0.0], "D": [0.05]}, index=idx)
    # コスト無し：D ロング(+0.05) − A ショート(−0.05) = 0.10
    r0 = long_short_returns(factor, fwd, quantile=0.25, min_names=4)
    assert r0.iloc[0] == pytest.approx(0.10)
    # コスト有り：初回は全建てで回転=2、片道10bps → 0.002 控除
    r1 = long_short_returns(factor, fwd, quantile=0.25, costs_bps=10, min_names=4)
    assert r1.iloc[0] == pytest.approx(0.10 - 0.002)


# --- 1-2 上場廃止リターン補完（生存者バイアス） -----------------------------
def test_delisting_mask_detects_exit_not_panel_end():
    idx = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    # A=2月で退出（3月NaN）、B=終端まで在籍（廃止でない）
    price = pd.DataFrame({"A": [100.0, 50.0, np.nan], "B": [10.0, 11.0, 12.0]}, index=idx)
    m = delisting_mask(price)
    assert bool(m.loc[idx[1], "A"]) is True            # 最終取引月=2月を廃止月に
    assert bool(m["B"].any()) is False                 # 終端在籍は廃止でない


def test_impute_delisting_policies():
    idx = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    price = pd.DataFrame({"A": [100.0, 50.0, np.nan], "B": [10.0, 11.0, 12.0]}, index=idx)
    base = forward_returns(price)
    assert np.isnan(base.loc[idx[1], "A"])             # 素では廃止月が NaN＝バイアス源
    lp = impute_delisting(base, price, policy="last_price")
    assert lp.loc[idx[1], "A"] == 0.0                  # 最終気配清算＝増分0（偽損失なし）
    m100 = impute_delisting(base, price, policy="all_minus100")
    assert m100.loc[idx[1], "A"] == -1.0              # 全損下限
    heur = impute_delisting(base, price, policy="heuristic", heuristic_threshold=-0.4)
    assert heur.loc[idx[1], "A"] == -1.0             # 直近 -50% < -40% → 倒産プロキシ
    assert lp.loc[idx[0], "B"] == pytest.approx(0.1)  # 通常銘柄は不変


# --- 1-3 配当込みトータルリターン ------------------------------------------
def test_total_forward_returns_adds_dividend_yield():
    idx = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    adj = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)   # +10%/月
    raw = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    dps = pd.DataFrame({"A": [12.0, 12.0, 12.0]}, index=idx)      # 年12円→月1円
    tr = total_forward_returns(adj, raw, dps, periods_per_year=12)
    assert tr.loc[idx[0], "A"] == pytest.approx(0.10 + 1.0 / 100.0)  # 価格0.10＋配当0.01
    # dps 無し＝価格リターンのみ（後方互換＝forward_returns）
    assert total_forward_returns(adj).loc[idx[0], "A"] == pytest.approx(0.10)


# --- Part3-1 流動性ユニバース（絶対しきい値・PIT） --------------------------
def test_liquid_universe_mask_thresholds():
    idx = pd.to_datetime(["2025-01-31"])
    close = pd.DataFrame({"A": [150.0], "B": [80.0], "C": [200.0]}, index=idx)
    mcap = pd.DataFrame({"A": [2e10], "B": [5e10], "C": [5e9]}, index=idx)
    adv = pd.DataFrame({"A": [1e8], "B": [1e8], "C": [1e8]}, index=idx)
    m = liquid_universe_mask(close, mcap, adv, min_price=100, min_mcap=1e10, min_adv=5e7)
    assert bool(m.loc[idx[0], "A"]) is True            # 全条件 OK
    assert bool(m.loc[idx[0], "B"]) is False           # 株価 80 < 100
    assert bool(m.loc[idx[0], "C"]) is False           # 時価総額 5e9 < 1e10


def test_liquid_universe_presets_smallcap_30b():
    from invest_system.equities.universe import LIQUID_UNIVERSE_PRESETS, liquid_universe_mask
    cfg = LIQUID_UNIVERSE_PRESETS["smallcap_30b"]
    assert cfg["min_mcap"] == 3e9
    idx = pd.to_datetime(["2025-01-31"])
    close = pd.DataFrame({"A": [150.0]}, index=idx)
    mcap = pd.DataFrame({"A": [4e9]}, index=idx)       # ¥40億＝30B床で OK、10B床で NG
    adv = pd.DataFrame({"A": [1e8]}, index=idx)
    m_sc = liquid_universe_mask(close, mcap, adv, **cfg)
    m_prod = liquid_universe_mask(close, mcap, adv, **LIQUID_UNIVERSE_PRESETS["production"])
    assert bool(m_sc.loc[idx[0], "A"]) is True
    assert bool(m_prod.loc[idx[0], "A"]) is False


# --- Part2 正準ポリシー（外れ値・欠損） ------------------------------------
def test_winsorize_cross_sectional_clips_tails():
    from invest_system.equities.factors import winsorize_cross_sectional
    df = pd.DataFrame({"a": [0.0], "b": [1.0], "c": [2.0], "d": [100.0]},
                      index=pd.to_datetime(["2025-01-31"]))
    w = winsorize_cross_sectional(df, lower=0.0, upper=0.9).iloc[0]
    assert w["d"] < 100.0 and w["a"] == 0.0            # 上側外れ値をクリップ


def test_rank_and_fill_neutralizes_missing():
    from invest_system.equities.factors import rank_and_fill
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [np.nan]},
                      index=pd.to_datetime(["2025-01-31"]))
    rf = rank_and_fill(df).iloc[0]
    assert rf["c"] == 0.0                               # 欠損は中立0（方向バイアスなし）
    assert rf["a"] == pytest.approx(-1.0) and rf["b"] == pytest.approx(1.0)


# --- Part2-9 結合パネルの一体先読み不変（feature×label） --------------------
def test_joint_feature_label_no_lookahead():
    # feature 側：開示 long → as-of（≤t のみ採用、未来開示の改変に不変）
    long = pd.DataFrame([
        {"Code": "A", "DiscDate": "2024-06-20", "period_end": "2024-03-31", "total_assets": 100.0},
        {"Code": "A", "DiscDate": "2025-06-20", "period_end": "2025-03-31", "total_assets": 121.0},
    ])
    months = pd.to_datetime(["2025-01-31", "2025-02-28"])
    feat = point_in_time(long, months, ["total_assets"], lag_days=1)["total_assets"]
    assert feat.loc["2025-01-31", "A"] == 100.0        # 2025 開示はまだ見えない
    mut = long.copy()
    mut.loc[mut["DiscDate"] == "2025-06-20", "total_assets"] = 9999.0
    feat2 = point_in_time(mut, months, ["total_assets"], lag_days=1)["total_assets"]
    assert feat2.loc["2025-01-31", "A"] == 100.0       # 未来開示改変に不変

    # label 側：total_forward_returns は price[t],price[t+1],dps[t] のみ＝t+2 改変で不変
    idx = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    adj = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    dps = pd.DataFrame({"A": [12.0, 12.0, 12.0]}, index=idx)
    lab = total_forward_returns(adj, adj.copy(), dps)
    adj2 = adj.copy()
    adj2.loc[idx[2]] = 999.0                            # t+2（3月）を改変
    lab2 = total_forward_returns(adj2, adj.copy(), dps)
    assert lab.loc[idx[0], "A"] == pytest.approx(lab2.loc[idx[0], "A"])  # label[t] は t→t+1 のみ
