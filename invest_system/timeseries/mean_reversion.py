"""時系列ミーンリバージョンの検定（柱D・KB §11.1）。

平均回帰するか否かを建玉前にゲートする純関数群。statsmodels は関数内 lazy import
（frac_diff.py の慣習）。すべて「渡された ≤t 系列」で因果的に計算＝先読みなし（DP12）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(x) -> np.ndarray:
    """Series/array → float の1次元 ndarray（NaN 除去）。"""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.to_numpy(dtype=float).ravel()
    a = np.asarray(x, dtype=float).ravel()
    return a[~np.isnan(a)]


def _ou_fit(s: np.ndarray) -> tuple[float, float]:
    """ΔS_t = a + b·S_{t-1} を OLS し (a, b) を返す。"""
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
    return float(coef[0]), float(coef[1])


def half_life(spread) -> float:
    """OU/AR(1) 半減期（バー数）。ΔS=a+b·S_lag を OLS、φ=1+b、hl=-ln2/ln(φ)。

    φ≥1（非回帰＝トレンド/発散）→ inf。φ≤0（過回帰/振動）は下限クリップ＝高速回帰。
    KB §11.1 / DP14（保有ホライズン・再校正の導出器）。
    """
    s = _clean(spread)
    if s.size < 3 or np.std(s) == 0:
        return float("inf")
    _, b = _ou_fit(s)
    phi = 1.0 + b
    if phi >= 1.0 or not np.isfinite(phi):
        return float("inf")
    phi = max(phi, 1e-6)
    return float(-np.log(2.0) / np.log(phi))


def ou_params(spread) -> tuple[float, float, float]:
    """OU 母数 (theta=平均回帰速度, mu=長期平均, sigma=拡散)。推定不能は nan。"""
    s = _clean(spread)
    if s.size < 4 or np.std(s) == 0:
        return (float("nan"), float("nan"), float("nan"))
    a, b = _ou_fit(s)
    theta = -b
    mu = (-a / b) if b != 0 else float("nan")
    resid = np.diff(s) - (a + b * s[:-1])
    sigma = float(np.std(resid, ddof=1)) if resid.size > 2 else float("nan")
    return (float(theta), float(mu), sigma)


def hurst_exponent(series, max_lag: int = 100) -> float:
    """Hurst 指数。lagged-diff の標準偏差 ∝ lag^H の傾き。

    H<0.5 平均回帰／H≈0.5 ランダムウォーク／H>0.5 トレンド。KB §11.1。
    """
    x = _clean(series)
    if x.size < 20:
        return float("nan")
    hi = min(max_lag, x.size // 2)
    if hi < 4:
        return float("nan")
    lags = np.arange(2, hi)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    ok = tau > 0
    if ok.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
    return float(slope)


def variance_ratio(series, lags=(2, 4, 8, 16)) -> pd.Series:
    """Lo-MacKinlay 分散比 VR(q)=Var(q期変化)/(q·Var(1期変化))。

    RW で≈1、VR<1 平均回帰、VR>1 トレンド。返り値 index=q。KB §11.1。
    """
    x = _clean(series)
    ret = np.diff(x)
    n = ret.size
    var1 = np.var(ret, ddof=1) if n > 1 else 0.0
    out = {}
    for q in lags:
        q = int(q)
        if q < 2 or n < 2 * q or var1 == 0:
            out[q] = float("nan")
            continue
        qret = np.convolve(ret, np.ones(q), mode="valid")
        out[q] = float(np.var(qret, ddof=1) / (q * var1))
    return pd.Series(out, name="variance_ratio")


def adf_pvalue(series, regression: str = "c") -> float:
    """ADF 単位根検定の p 値（小さいほど定常＝平均回帰）。statsmodels lazy import。"""
    from statsmodels.tsa.stattools import adfuller

    x = _clean(series)
    if x.size < 10 or np.std(x) == 0:
        return float("nan")
    return float(adfuller(x, regression=regression, autolag="AIC")[1])
