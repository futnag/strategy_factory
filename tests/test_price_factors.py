"""価格系・流動性系ファクター（GKX Phase 1）：先読み不変・整合・符号・境界（ネット不要）。

合成フィクスチャで純関数を検証。handoff §3 の PIT 先読み不変を assert、既存材化（vol_20・
momentum_12_1・reversal）との整合、符号規約（大きいほどロング側）、ベクトル化βの正しさ。
"""
import numpy as np
import pandas as pd

from invest_system.equities import price_factors as pf


def _prices(n: int = 320, cols=("A", "B", "C"), seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, size=(n, len(cols)))
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=idx, columns=list(cols))


def test_month_constant():
    assert pf.MONTH == 21


def test_momentum_and_reversal_formulas():
    px = _prices()
    m3 = pf.mom_3m(px)
    exp = px.shift(21) / px.shift(63) - 1.0
    assert np.allclose(m3.dropna().values, exp.dropna().values)
    rev = pf.mom_1m_reversal(px)
    assert np.allclose(rev.dropna().values, (-(px / px.shift(21) - 1.0)).dropna().values)


def test_long_term_reversal_skips_recent_12m():
    px = _prices(n=900)                                 # 36M=756 営業日 + バッファ
    ltr = pf.mom_36m_reversal(px)
    exp = -(px.shift(12 * pf.MONTH) / px.shift(36 * pf.MONTH) - 1.0)
    assert np.allclose(ltr.dropna().values, exp.dropna().values)
    # 直近 12M スキップ＝最新月の価格を改変しても最新断面の値は不変（短期リバーサルと非重複）
    px2 = px.copy()
    px2.iloc[-pf.MONTH:] *= 1.3
    last = ltr.dropna().index[-1]
    assert np.allclose(pf.mom_36m_reversal(px).loc[last].values,
                       pf.mom_36m_reversal(px2).loc[last].values, equal_nan=True)


def test_realized_vol_matches_feature_store_vol_negated():
    # 既存 feature_store の vol_20 = ret.rolling(20).std()*sqrt(252)。新 realized_vol は同式の負号。
    px = _prices()
    ret = px.pct_change()
    exp = -(ret.rolling(20, min_periods=10).std() * np.sqrt(252.0))
    got = pf.realized_vol(px, 20)
    deep = slice(100, None)                              # 窓が十分埋まる領域で比較（min_periods差を回避）
    assert np.allclose(got.iloc[deep].values, exp.iloc[deep].values, equal_nan=True)
    assert (got.dropna().values <= 1e-12).all()         # 低ボラ=負号＝非正


def test_lookahead_invariance():
    px = _prices()
    t = px.index[200]
    px2 = px.copy()
    px2.iloc[260:] *= 1.5                               # t より未来だけ改変
    mkt = px.pct_change().mean(axis=1)
    mkt2 = px2.pct_change().mean(axis=1)
    for fn in (lambda p: pf.mom_3m(p), lambda p: pf.realized_vol(p, 60),
               lambda p: pf.max_return(p), lambda p: pf.zero_return_days(p, 60)):
        a, b = fn(px), fn(px2)
        assert np.allclose(a.loc[t].values, b.loc[t].values, equal_nan=True)
    # 市場系列を使うβ/ivol も ≤t 不変（市場も未来だけ変わる）
    for fn in (pf.market_beta, pf.idiosyncratic_vol):
        a, b = fn(px, mkt, 120), fn(px2, mkt2, 120)
        assert np.allclose(a.loc[t].values, b.loc[t].values, equal_nan=True)


def test_beta_recovers_known_and_ivol_zero_for_exact():
    # stockA=1.5×market, stockB=0.5×market（厳密線形）→ β=1.5/0.5、ivol≈0。
    idx = pd.bdate_range("2020-01-01", periods=260)
    rng = np.random.default_rng(1)
    mkt = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    ret = pd.DataFrame({"A": 1.5 * mkt, "B": 0.5 * mkt})
    px = (1.0 + ret).cumprod() * 100.0
    beta = pf.market_beta(px, mkt, window=120)
    ivol = pf.idiosyncratic_vol(px, mkt, window=120)
    last = beta.index[-1]
    assert abs(beta.loc[last, "A"] - (-1.5)) < 1e-6     # 低ベータ=負号
    assert abs(beta.loc[last, "B"] - (-0.5)) < 1e-6
    assert abs(ivol.loc[last, "A"]) < 1e-6              # 厳密線形＝特異ボラ0
    assert abs(ivol.loc[last, "B"]) < 1e-6


def test_industry_momentum_broadcast_within_sector():
    px = _prices(cols=("A", "B", "C"))
    sector = pd.Series({"A": "X", "B": "X", "C": "Y"})
    im = pf.industry_momentum(px, sector)
    last = im.dropna().index[-1]
    assert im.loc[last, "A"] == im.loc[last, "B"]       # 同一業種は同値
    assert im.loc[last, "A"] != im.loc[last, "C"]       # 別業種は別値


def test_liquidity_signs_and_bounds():
    px = _prices()
    rng = np.random.default_rng(2)
    va = pd.DataFrame(rng.uniform(1e8, 1e9, size=px.shape), index=px.index, columns=px.columns)
    amihud = pf.amihud_illiquidity(px, va, 60)
    assert (amihud.dropna().values >= 0).all()          # 非流動性=正符号
    z = pf.zero_return_days(px, 60)
    vals = z.dropna().values
    assert ((vals >= 0) & (vals <= 1)).all()            # 比率 [0,1]
    # dollar_volume：Va が大きい銘柄ほど小さい値（大型=ショート側＝負号）
    va2 = va.copy(); va2["A"] *= 10.0
    dv = pf.dollar_volume(va2, 60)
    last = dv.dropna().index[-1]
    assert dv.loc[last, "A"] < dv.loc[last, "B"]


def test_turnover_sign_and_compute_all_keys():
    px = _prices()
    va = pd.DataFrame(5e8, index=px.index, columns=px.columns)
    mcap = pd.DataFrame(1e12, index=px.index, columns=px.columns)
    tov = pf.turnover(va, mcap, 60)
    assert (tov.dropna().values <= 0).all()             # 高回転=低リターン＝負号
    out = pf.compute_all(px, va=va, mcap=mcap, sector=pd.Series({"A": "X", "B": "X", "C": "Y"}))
    for k in ("mom_3m", "rvol_60", "ivol", "beta", "max_ret", "ret_skew",
              "amihud_illiq", "dollar_volume", "zero_ret_days", "turnover",
              "industry_momentum"):
        assert k in out and out[k].shape == px.shape
