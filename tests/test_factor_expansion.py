"""特徴量拡充 A（信用/空売り・微細構造）＋ B（fins_summary 新ファンダ）の検証（ネット不要）。

handoff §5 受け入れ基準：①先読み不変②公表日アンカーPIT（基準日でなく公表日・未来漏れ無し）
③微細構造の調整済OHLC（分割日で発散しない）④SUE のCurFYEn連携⑤季節性の最低年数。
合成フィクスチャで純関数を検証。
"""
import numpy as np
import pandas as pd

from invest_system.equities import fundamental_factors as ff
from invest_system.equities import fundamentals as fu
from invest_system.equities import holdings_factors as hf
from invest_system.features import microstructure as ms


# === A：公表日アンカー ======================================================
def test_shift_business_days_plus2():
    cal = pd.bdate_range("2016-06-01", "2016-07-31")        # 平日カレンダー
    fri = pd.Timestamp("2016-06-17")                        # 金曜
    av = hf.shift_business_days([fri], 2, cal).iloc[0]
    assert av == pd.Timestamp("2016-06-21")                 # 月=+1, 火=+2（翌週第2営業日）


def test_weekly_margin_publication_anchor():
    cal = pd.bdate_range("2016-06-01", "2016-07-31")
    weekly = pd.DataFrame({"Date": ["2016-06-17", "2016-06-17"], "Code": ["A", "B"],
                           "LongVol": [100.0, 200.0], "ShrtVol": [50.0, 50.0]})
    wl = hf.weekly_margin_long(weekly, cal)
    assert (wl["available_date"] == pd.Timestamp("2016-06-21")).all()   # 基準金曜+2営業日
    # margin_imbalance = (Long-Short)/(Long+Short)
    a = wl[wl["Code"] == "A"].iloc[0]
    assert abs(a["margin_imbalance"] - (100 - 50) / 150) < 1e-9


def test_short_position_uses_discdate_not_calcdate():
    pos = pd.DataFrame({"DiscDate": ["2016-06-14"], "CalcDate": ["2016-06-13"],
                        "Code": ["A"], "ShrtPosToSO": [0.03], "ShrtPosShares": [1000.0]})
    sl = hf.short_position_long(pos)
    assert sl.iloc[0]["available_date"] == pd.Timestamp("2016-06-14")   # 開示日（≠基準日13日）
    assert sl.iloc[0]["short_interest"] == 0.03


def test_holdings_no_future_leak():
    # available_date アンカーで point_in_time に流すと、未来の公表は ≤t に漏れない
    cal = pd.bdate_range("2016-06-01", "2016-08-31")
    weekly = pd.DataFrame({"Date": ["2016-06-17", "2016-07-15"], "Code": ["A", "A"],
                           "LongVol": [100.0, 999.0], "ShrtVol": [50.0, 50.0]})
    wl = hf.weekly_margin_long(weekly, cal)
    me = pd.to_datetime(["2016-06-30", "2016-07-31"])
    pan = fu.point_in_time(wl, me, ["margin_imbalance"], date_col="available_date",
                           code_col="Code", lag_days=0)["margin_imbalance"]
    v_june = pan.loc["2016-06-30", "A"]                      # 6/17 分(+2営業日=6/21)が見える
    # 7/15 の残高(LongVol=999)を改変しても 6 月末 as-of は不変
    weekly2 = weekly.copy(); weekly2.loc[1, "LongVol"] = 1.0
    wl2 = hf.weekly_margin_long(weekly2, cal)
    pan2 = fu.point_in_time(wl2, me, ["margin_imbalance"], date_col="available_date",
                            code_col="Code", lag_days=0)["margin_imbalance"]
    assert pan.loc["2016-06-30", "A"] == pan2.loc["2016-06-30", "A"]


# === A：微細構造（調整済OHLCの必要性） ======================================
def test_roll_spread_diverges_on_split_so_use_adjusted():
    # 分割(終値が半分にジャンプ)を含む生終値は roll_spread が発散。調整済(連続)は発散しない。
    idx = pd.bdate_range("2020-01-01", periods=60)
    base = pd.Series(np.linspace(100, 101, 60), index=idx)   # ほぼ平坦
    raw = base.copy(); raw.iloc[30:] /= 2.0                  # 30 日目に 2:1 分割（生は半値）
    rs_raw = ms.roll_spread(raw, 20)
    rs_adj = ms.roll_spread(base, 20)                        # 調整済(連続)
    # 分割直後の窓で生は大きく、調整済は小さい
    assert rs_raw.iloc[31:50].max() > 5 * (rs_adj.iloc[31:50].max() + 1e-9)


# === B：SUE の CurFYEn 連携 =================================================
def _fins_fixture():
    rows = [
        # A 社 FY2024（前年・成長の分母）
        {"DiscDate": "2024-05-10", "Code": "A", "CurPerType": "FY", "CurFYEn": "2024-03-31",
         "EPS": 100.0, "FEPS": np.nan, "Sales": 1000.0, "NP": 80.0, "Eq": 500.0, "OP": 120.0,
         "PayoutRatioAnn": 0.30},
        # A 社 FY2025：1Q/2Q/3Q 予想 → FY 実績
        {"DiscDate": "2024-08-01", "Code": "A", "CurPerType": "1Q", "CurFYEn": "2025-03-31",
         "EPS": 20.0, "FEPS": 100.0, "Sales": np.nan, "NP": np.nan, "Eq": np.nan, "OP": np.nan,
         "PayoutRatioAnn": np.nan},
        {"DiscDate": "2024-11-01", "Code": "A", "CurPerType": "2Q", "CurFYEn": "2025-03-31",
         "EPS": 55.0, "FEPS": 110.0, "Sales": np.nan, "NP": np.nan, "Eq": np.nan, "OP": np.nan,
         "PayoutRatioAnn": np.nan},
        {"DiscDate": "2025-02-01", "Code": "A", "CurPerType": "3Q", "CurFYEn": "2025-03-31",
         "EPS": 95.0, "FEPS": 120.0, "Sales": np.nan, "NP": np.nan, "Eq": np.nan, "OP": np.nan,
         "PayoutRatioAnn": np.nan},
        {"DiscDate": "2025-05-10", "Code": "A", "CurPerType": "FY", "CurFYEn": "2025-03-31",
         "EPS": 130.0, "FEPS": np.nan, "Sales": 1100.0, "NP": 88.0, "Eq": 550.0, "OP": 132.0,
         "PayoutRatioAnn": 0.30},
    ]
    return pd.DataFrame(rows)


def test_sue_linkage_recent_initial_and_revision():
    out_fy, out_rev = ff.disclosure_features(_fins_fixture())
    fy25 = out_fy[out_fy["DiscDate"] == pd.Timestamp("2025-05-10")].iloc[0]
    assert abs(fy25["sue_recent_raw"] - (130.0 - 120.0)) < 1e-9    # 実績−直近(3Q)予想
    assert abs(fy25["sue_initial_raw"] - (130.0 - 100.0)) < 1e-9   # 実績−期初(1Q)予想
    assert abs(fy25["sales_growth"] - 0.10) < 1e-9                 # 1100/1000−1
    assert abs(fy25["profit_growth"] - 0.10) < 1e-9
    # 持続可能成長 = ROE×(1−payout) = (88/550)×(1−0.30)
    assert abs(fy25["sustainable_growth"] - (88.0 / 550.0) * 0.70) < 1e-9
    # 予想改訂：100→110→120 の差（+10, +10）
    revs = out_rev[out_rev["Code"] == "A"]["forecast_revision_raw"].tolist()
    assert revs == [10.0, 10.0]


def test_stability_negative_sign():
    # ROE の trailing 標準偏差に負号（≥3点で発火）。4 期 ROE を与える。
    rows = []
    for i, (eps, eq) in enumerate([(80, 500), (90, 500), (70, 500), (100, 500)]):
        y = 2022 + i
        rows.append({"DiscDate": f"{y+1}-05-10", "Code": "A", "CurPerType": "FY",
                     "CurFYEn": f"{y+1}-03-31", "EPS": eps, "FEPS": np.nan, "Sales": 1000.0,
                     "NP": eps, "Eq": eq, "OP": 100.0, "PayoutRatioAnn": 0.3})
    out_fy, _ = ff.disclosure_features(pd.DataFrame(rows))
    last = out_fy.sort_values("DiscDate").iloc[-1]
    assert last["roe_stability"] <= 0                              # 負号（安定度=−ボラ）


# === B：52週高値・季節性・Dimson β =========================================
def _prices(n=900, cols=("A", "B"), seed=0):
    idx = pd.bdate_range("2016-01-01", periods=n)
    rng = np.random.default_rng(seed)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(n, len(cols))), axis=0))
    return pd.DataFrame(px, index=idx, columns=list(cols))


def test_high_52w_bounds_and_lookahead():
    px = _prices()
    h = ff.high_52w(px)
    v = h.dropna().values
    assert ((v > 0) & (v <= 1.0 + 1e-9)).all()                    # (0,1]
    t = h.dropna().index[len(h.dropna()) // 2]
    px2 = px.copy(); px2.loc[t + pd.Timedelta(days=1):] *= 2.0     # 未来改変
    assert np.allclose(ff.high_52w(px).loc[t].values,
                       ff.high_52w(px2).loc[t].values, equal_nan=True)


def test_seasonality_min_years():
    # 月次 6 年分 → 最低5年要求で「6年目以降」のみ値（同暦月の過去観測≥5）。
    idx = pd.bdate_range("2016-01-01", periods=1600)              # ~6.3年
    px = pd.DataFrame(100.0 * np.exp(np.cumsum(
        np.random.default_rng(0).normal(0, 0.01, size=(len(idx), 1)), axis=0)),
        index=idx, columns=["A"])
    s = ff.seasonality(px, min_years=5)
    early = s.loc[:"2020-12-31", "A"].notna().sum()               # 5年未満の期間
    late = s.loc["2021-06-30":, "A"].notna().sum()
    assert early == 0 and late >= 1                               # 最低5年の床が効く


def test_dimson_beta_sum_of_lagged():
    px = _prices(n=400, cols=("A",))
    mkt = px["A"].pct_change()                                    # 任意の市場系列
    from invest_system.equities.price_factors import _rolling_beta_residvar
    db = ff.dimson_beta(px, mkt, window=120, lags=1)
    b0, _ = _rolling_beta_residvar(px.pct_change(), mkt, 120)
    b1, _ = _rolling_beta_residvar(px.pct_change(), mkt.shift(1), 120)
    last = db.dropna().index[-1]
    assert abs(db.loc[last, "A"] - (-(b0.loc[last, "A"] + b1.loc[last, "A"]))) < 1e-9
