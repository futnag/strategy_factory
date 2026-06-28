"""レジーム検知アルゴリズム（ウォークフォワード・先読みなし）。

各手法は時点 t で ≤t のデータのみを用いてレジームラベル/確率を出力する。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import (
    CPD_METHODS,
    DEFAULT_REFIT_EVERY,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_WARMUP_WEEKS,
    GMM_N_COMPONENTS,
    HMM_N_STATES,
    RANDOM_STATE,
)


@dataclass
class RegimeOutput:
    """1手法のウォークフォワード出力。"""
    method: str
    sector: str
    regime: pd.Series          # 離散ラベル（整数）
    regime_prob: pd.Series     # 主レジーム確率（連続、0-1）
    n_states: int = 2
    meta: dict = field(default_factory=dict)


def _slice_window(X: np.ndarray, t: int, window: Optional[int]) -> np.ndarray:
    """ローリング or 拡張窓で ≤t のデータを返す。"""
    if window is None:
        return X[: t + 1]
    start = max(0, t + 1 - window)
    return X[start : t + 1]


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """窓内で標準化（平均・分散は当該窓のみ）。"""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _high_vol_state_index(states: np.ndarray, vol_proxy: np.ndarray) -> int:
    """高ボラ状態のインデックスを推定（経済的解釈の統一）。"""
    uniq = np.unique(states)
    means = {s: vol_proxy[states == s].mean() for s in uniq if (states == s).any()}
    return int(max(means, key=means.get)) if means else 0


# ------------------------------------------------------------------ HMM
def walk_forward_hmm(
    features: pd.DataFrame,
    *,
    sector: str = "",
    n_states: int = 2,
    warmup: int = DEFAULT_WARMUP_WEEKS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    window: Optional[int] = DEFAULT_ROLLING_WINDOW,
    random_state: int = RANDOM_STATE,
) -> RegimeOutput:
    """Gaussian HMM のウォークフォワード推定。"""
    from hmmlearn.hmm import GaussianHMM

    idx = features.index
    X = features.values.astype(float)
    n = len(X)
    regime = pd.Series(np.nan, index=idx)
    prob = pd.Series(np.nan, index=idx)
    model = None

    for t in range(warmup, n):
        if t < warmup:
            continue
        win = _slice_window(X, t, window)
        win, _, _ = _standardize_fit(win)
        if len(win) < warmup:
            continue
        try:
            if model is None or (t - warmup) % refit_every == 0:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=random_state,
                    tol=1e-3,
                )
                model.fit(win)
            states = model.predict(win)
            post = model.predict_proba(win)
            regime.iloc[t] = states[-1]
            # 高ボラ状態の確率を regime_prob として保存
            vol_col = 0
            if "realized_vol_w" in features.columns:
                vol_col = list(features.columns).index("realized_vol_w")
            hv = _high_vol_state_index(states, win[:, vol_col])
            prob.iloc[t] = post[-1, hv]
        except Exception:  # noqa: BLE001
            continue

    return RegimeOutput(
        method=f"hmm_{n_states}",
        sector=sector,
        regime=regime,
        regime_prob=prob,
        n_states=n_states,
    )


# ------------------------------------------------------------------ Change Point
def _cpd_regime_from_breaks(n: int, breaks: list[int], n_states: int = 2) -> np.ndarray:
    """変化点リストから区間ラベルを生成（交互ラベル）。"""
    labels = np.zeros(n, dtype=int)
    if not breaks:
        return labels
    seg = 0
    prev = 0
    pts = sorted(set(b for b in breaks if 0 < b < n))
    for b in pts + [n]:
        labels[prev:b] = seg % n_states
        seg += 1
        prev = b
    return labels


def walk_forward_cpd(
    features: pd.DataFrame,
    *,
    sector: str = "",
    algo: str = "pelt",
    warmup: int = DEFAULT_WARMUP_WEEKS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    window: Optional[int] = DEFAULT_ROLLING_WINDOW,
    pen: float = 3.0,
    n_states: int = 2,
) -> RegimeOutput:
    """ruptures による変化点検知のウォークフォワード版。"""
    import ruptures as rpt

    idx = features.index
    X = features.values.astype(float)
    n = len(X)
    regime = pd.Series(np.nan, index=idx)
    prob = pd.Series(np.nan, index=idx)
    last_breaks: list[int] = []

    for t in range(warmup, n):
        win = _slice_window(X, t, window)
        win, _, _ = _standardize_fit(win)
        if len(win) < warmup:
            continue
        try:
            if (t - warmup) % refit_every != 0 and last_breaks:
                states = _cpd_regime_from_breaks(len(win), last_breaks, n_states)
            else:
                y = win.reshape(-1, 1)
                min_size = max(10, len(win) // 20)
                if algo == "pelt":
                    algo_obj = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(y)
                    bk = algo_obj.predict(pen=pen)
                elif algo == "binseg":
                    algo_obj = rpt.Binseg(model="rbf", min_size=min_size, jump=1).fit(y)
                    bk = algo_obj.predict(n_bkps=min(3, len(win) // min_size))
                elif algo == "bottomup":
                    algo_obj = rpt.BottomUp(model="rbf", min_size=min_size, jump=1).fit(y)
                    bk = algo_obj.predict(n_bkps=min(3, len(win) // min_size))
                else:
                    raise ValueError(f"unknown cpd algo: {algo}")
                last_breaks = [b for b in bk if b < len(win)]
                states = _cpd_regime_from_breaks(len(win), last_breaks, n_states)
            regime.iloc[t] = states[-1]
            # レジーム確信度: 現在区間の長さに基づくヒューリスティック
            seg_len = 1
            for b in reversed(last_breaks):
                if b < len(win):
                    seg_len = len(win) - b
                    break
            prob.iloc[t] = min(1.0, seg_len / 20.0)
        except Exception:  # noqa: BLE001
            continue

    return RegimeOutput(
        method=f"cpd_{algo}",
        sector=sector,
        regime=regime,
        regime_prob=prob,
        n_states=n_states,
    )


# ------------------------------------------------------------------ GMM
def walk_forward_gmm(
    features: pd.DataFrame,
    *,
    sector: str = "",
    n_components: int = 2,
    warmup: int = DEFAULT_WARMUP_WEEKS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    window: Optional[int] = DEFAULT_ROLLING_WINDOW,
    random_state: int = RANDOM_STATE,
) -> RegimeOutput:
    """ローリング GMM クラスタリング（特徴量空間）。"""
    from sklearn.mixture import GaussianMixture

    idx = features.index
    X = features.values.astype(float)
    n = len(X)
    regime = pd.Series(np.nan, index=idx)
    prob = pd.Series(np.nan, index=idx)
    gmm = None

    for t in range(warmup, n):
        win = _slice_window(X, t, window)
        win, _, _ = _standardize_fit(win)
        if len(win) < warmup:
            continue
        try:
            if gmm is None or (t - warmup) % refit_every == 0:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="diag",
                    random_state=random_state,
                    max_iter=200,
                    n_init=3,
                )
                gmm.fit(win)
            labels = gmm.predict(win)
            post = gmm.predict_proba(win)
            regime.iloc[t] = labels[-1]
            vol_col = 0
            if "realized_vol_w" in features.columns:
                vol_col = list(features.columns).index("realized_vol_w")
            hv = _high_vol_state_index(labels, win[:, vol_col])
            prob.iloc[t] = post[-1, hv]
        except Exception:  # noqa: BLE001
            continue

    return RegimeOutput(
        method=f"gmm_{n_components}",
        sector=sector,
        regime=regime,
        regime_prob=prob,
        n_states=n_components,
    )


# ------------------------------------------------------------------ Markov Switching
def walk_forward_markov_switching(
    features: pd.DataFrame,
    *,
    sector: str = "",
    n_regimes: int = 2,
    warmup: int = DEFAULT_WARMUP_WEEKS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    window: Optional[int] = None,
) -> RegimeOutput:
    """statsmodels MarkovRegression のフィルタ確率（因果）。"""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    idx = features.index
    # ボラをスイッチング対象に（リターン系列）
    y_col = "log_return_w" if "log_return_w" in features.columns else features.columns[0]
    y = (features[y_col] * 100.0).astype(float)
    n = len(y)
    regime = pd.Series(np.nan, index=idx)
    prob = pd.Series(np.nan, index=idx)
    params = None
    turb_idx = 0

    for t in range(warmup, n):
        win_y = y.iloc[: t + 1] if window is None else y.iloc[max(0, t + 1 - window) : t + 1]
        if len(win_y) < warmup:
            continue
        try:
            if params is None or (t - warmup) % refit_every == 0:
                mod = MarkovRegression(
                    win_y.values,
                    k_regimes=n_regimes,
                    trend="c",
                    switching_variance=True,
                )
                res = mod.fit(em_iter=50, search_reps=5, disp=False)
                params = res.params
                fp = np.asarray(res.filtered_marginal_probabilities)
                if fp.shape[0] != len(win_y):
                    fp = fp.T
                vol_proxy = features["realized_vol_w"].iloc[: t + 1].values[-len(win_y):]
                states = fp.argmax(axis=1)
                turb_idx = _high_vol_state_index(states, vol_proxy)
            else:
                mod = MarkovRegression(
                    win_y.values,
                    k_regimes=n_regimes,
                    trend="c",
                    switching_variance=True,
                )
                fp = np.asarray(mod.filter(params).filtered_marginal_probabilities)
                if fp.shape[0] != len(win_y):
                    fp = fp.T
            regime.iloc[t] = int(fp[-1].argmax())
            prob.iloc[t] = float(fp[-1, turb_idx])
        except Exception:  # noqa: BLE001
            continue

    return RegimeOutput(
        method=f"markov_{n_regimes}",
        sector=sector,
        regime=regime,
        regime_prob=prob,
        n_states=n_regimes,
    )


# ------------------------------------------------------------------ BOCPD
def walk_forward_bocpd(
    features: pd.DataFrame,
    *,
    sector: str = "",
    warmup: int = DEFAULT_WARMUP_WEEKS,
    hazard_mean: int = 52,
) -> RegimeOutput:
    """ベイズオンライン変化点検知（Adams-MacKay 型・因果）。"""
    from scipy.stats import t as student_t

    idx = features.index
    y_col = "log_return_w" if "log_return_w" in features.columns else features.columns[0]
    x = features[y_col].astype(float).values
    n = len(x)
    regime = pd.Series(np.nan, index=idx)
    prob = pd.Series(np.nan, index=idx)

    mu0, kappa0, alpha0 = 0.0, 1.0, 1.0
    beta0 = float(np.var(x[:warmup])) if warmup < n else 1.0
    H = 1.0 / hazard_mean
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    R = np.array([1.0])
    maps = np.zeros(n, dtype=int)
    cp_prob = np.zeros(n)

    for i in range(n):
        xi = x[i]
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        pred = student_t.pdf(xi, df=2 * alpha, loc=mu, scale=scale)
        growth = R * pred * (1.0 - H)
        cp = np.sum(R * pred * H)
        R = np.concatenate([[cp], growth])
        R /= R.sum()
        maps[i] = int(np.argmax(R))
        cp_prob[i] = float(cp)
        mu_n = (kappa * mu + xi) / (kappa + 1.0)
        kappa_n = kappa + 1.0
        alpha_n = alpha + 0.5
        beta_n = beta + (kappa * (xi - mu) ** 2) / (2.0 * (kappa + 1.0))
        mu = np.concatenate([[mu0], mu_n])
        kappa = np.concatenate([[kappa0], kappa_n])
        alpha = np.concatenate([[alpha0], alpha_n])
        beta = np.concatenate([[beta0], beta_n])
        if len(R) > 200:
            R = R[:200]
            mu, kappa, alpha, beta = mu[:200], kappa[:200], alpha[:200], beta[:200]
            R /= R.sum()

    # run-length からレジームラベル（変化点で交互切替）
    labels = np.zeros(n, dtype=int)
    state = 0
    prev_map = 0
    for i in range(n):
        if i > 0 and maps[i] < maps[i - 1] * 0.5:
            state = 1 - state
        labels[i] = state
        prev_map = maps[i]

    regime.iloc[:] = labels
    prob.iloc[:] = 1.0 - cp_prob
    regime.iloc[:warmup] = np.nan
    prob.iloc[:warmup] = np.nan

    return RegimeOutput(
        method="bocpd",
        sector=sector,
        regime=regime,
        regime_prob=prob,
        n_states=2,
    )


# ------------------------------------------------------------------ 全手法リスト
def all_method_specs() -> list[dict]:
    """比較対象の全手法仕様。"""
    specs: list[dict] = []
    for k in HMM_N_STATES:
        specs.append({"fn": walk_forward_hmm, "kwargs": {"n_states": k}})
    for algo in CPD_METHODS:
        specs.append({"fn": walk_forward_cpd, "kwargs": {"algo": algo}})
    for k in GMM_N_COMPONENTS:
        specs.append({"fn": walk_forward_gmm, "kwargs": {"n_components": k}})
    specs.append({"fn": walk_forward_markov_switching, "kwargs": {"n_regimes": 2}})
    specs.append({"fn": walk_forward_markov_switching, "kwargs": {"n_regimes": 3}})
    specs.append({"fn": walk_forward_bocpd, "kwargs": {}})
    return specs


def _filter_kwargs(fn: Callable, kwargs: dict) -> dict:
    """検知器関数が受け付ける引数のみに絞る（パイプライン共通 kwargs の誤渡し防止）。"""
    allowed = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def run_detector(
    fn: Callable,
    features: pd.DataFrame,
    *,
    sector: str,
    config_kwargs: Optional[dict] = None,
) -> RegimeOutput:
    """単一検知器を実行。"""
    kw = _filter_kwargs(fn, config_kwargs or {})
    return fn(features, sector=sector, **kw)