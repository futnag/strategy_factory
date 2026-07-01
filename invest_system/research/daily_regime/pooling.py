"""daily_regime: 流動性ティア partial pooling 推定（③・接続設計 §5.4）。

データ枯渇する日足ボラ流動性 HMM を、**流動性ティアで括ったプールで放出推定**し、銘柄別に
filter する。役割分担：プールは安定（放出＝calm/stressed クラスタの位置）、HMM が銘柄別の
速い一過性ストレスを遷移で検知。**pooling 強度 λ は固定**（探索しない＝二重過剰適合回避）。

per-stock（detector.walk_forward_t_hmm）はアブレーション対照として残す。本モジュールは
pooled 版と、両者を比較する安定性指標を提供する。

PIT：プール選抜は as-of t のティア（membership ≤t の直近行）、特徴は ≤t 窓のみ。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .detector import StudentTHMM, _min_dwell_filter, _standardize_past, fit_best, fit_t_mixture
from invest_system.research.sector_regime.detectors import _align_states


def fit_pooled_emissions(
    stacked_X: np.ndarray, *, n_states: int, dof: float,
    n_init: int, random_state: int, align_col: int,
):
    """ティア横断のスタック観測から放出（means/covars）をプール推定し、整列して返す。

    時間連続でないプールに HMM 遷移は当てず、**Student-t 混合で放出のみ**を推定する（高速）。
    """
    means, covars, _ = fit_t_mixture(stacked_X, n_states=n_states, dof=dof,
                                     n_init=n_init, random_state=random_state)
    order = _align_states(np.arange(n_states), means[:, align_col], ascending=True)
    perm = np.array(order)
    return means[perm].copy(), covars[perm].copy()


def _peers_asof(membership: pd.DataFrame, code: str, t: pd.Timestamp) -> tuple[str, list[str]]:
    """as-of t（≤t 直近行）で code と同ティアの銘柄群を返す。"""
    row = membership.loc[:t]
    if row.empty:
        return "", []
    last = row.iloc[-1]
    tier = last.get(code)
    if tier is None or (isinstance(tier, float) and np.isnan(tier)):
        return "", []
    peers = [c for c in membership.columns if last.get(c) == tier]
    return str(tier), peers


def walk_forward_pooled_t_hmm(
    target_code: str,
    feat_panel: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    *,
    n_states: int = 2,
    dof: float = 4.0,
    sticky_kappa: float = 10.0,
    min_dwell: int = 3,
    refit_every: int = 20,
    window: Optional[int] = 504,
    warmup: int = 252,
    align_col: int = 0,
    n_init: int = 2,
    random_state: int = 42,
    max_pool_codes: int = 40,
    log_tilt: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """pooled 版 walk-forward：放出はティアプール推定・遷移は銘柄別・終端 filtered。

    log_tilt=(n,K) を渡すと方式A（事前分布注入）を適用（multiscale 用）。None でベースライン。
    """
    target = feat_panel[target_code]
    idx = target.index
    Xt = target.to_numpy(dtype=float)
    n, d = Xt.shape
    K = n_states
    prob = np.full((n, K), np.nan)
    raw_state = np.full(n, -1, dtype=int)
    refit_flag = np.zeros(n, dtype=bool)
    pool_size = np.full(n, 0, dtype=int)
    model: Optional[StudentTHMM] = None

    for t in range(warmup, n):
        lo = 0 if window is None else max(0, t + 1 - window)
        win = _standardize_past(Xt[lo:t + 1])
        if len(win) < warmup or not np.isfinite(win).all():
            continue
        if model is None or (t - warmup) % refit_every == 0:
            tier, peers = _peers_asof(membership, target_code, idx[t])
            stacks = []
            for c in peers[:max_pool_codes]:
                if c not in feat_panel:
                    continue
                w = feat_panel[c].loc[:idx[t]]
                w = w.to_numpy(dtype=float)[-window:] if window else w.to_numpy(dtype=float)
                w = _standardize_past(w)
                if len(w) >= warmup and np.isfinite(w).all():
                    stacks.append(w)
            if len(stacks) >= 2:                                   # プール推定
                stacked = np.vstack(stacks)
                means, covars = fit_pooled_emissions(
                    stacked, n_states=K, dof=dof,
                    n_init=n_init, random_state=random_state, align_col=align_col)
                model = StudentTHMM(n_states=K, dof=dof, sticky_kappa=sticky_kappa,
                                    random_state=random_state)
                model.means_, model.covars_ = means, covars
                model.startprob_ = np.full(K, 1.0 / K)
                model.transmat_ = np.full((K, K), 0.1 / max(K - 1, 1))
                np.fill_diagonal(model.transmat_, 0.9)
                model.fit_transmat_only(win)                       # 遷移のみ銘柄別に
                pool_size[t] = len(stacks)
            else:                                                  # プール不成立→per-stock 退避
                model = fit_best(win, n_states=K, dof=dof, sticky_kappa=sticky_kappa,
                                 n_init=n_init, random_state=random_state)
                model.align_by(model.means_[:, align_col])
            refit_flag[t] = True
        if log_tilt is None:
            f = model.filtered_proba(win)[-1]
        else:
            f = model.filtered_proba_injected(win, log_tilt[lo:t + 1])[-1]   # 方式A 注入
        prob[t] = f
        raw_state[t] = int(np.argmax(f))

    valid = raw_state >= 0
    state_dwell = raw_state.copy()
    if valid.any():
        state_dwell[valid] = _min_dwell_filter(raw_state[valid], min_dwell)

    out = pd.DataFrame(index=idx)
    out["state_filtered"] = np.where(valid, state_dwell, np.nan)
    out["prob_stressed"] = prob[:, K - 1]
    for k in range(K):
        out[f"p_state{k}"] = prob[:, k]
    out["refit"] = refit_flag
    out["pool_size"] = pool_size
    return out


def walk_forward_pooled_multi(
    target_code: str,
    feat_panel: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    tilts: dict,
    *,
    n_states: int = 2, dof: float = 4.0, sticky_kappa: float = 10.0, min_dwell: int = 3,
    refit_every: int = 20, window: Optional[int] = 504, warmup: int = 252,
    align_col: int = 0, n_init: int = 2, random_state: int = 42, max_pool_codes: int = 40,
) -> dict:
    """**1回のプール推定**で複数 tilt の filtered を同時出力（daily-only と two-layer を1パスで）。

    tilts: {name: (n,K) log_tilt or None}。返り値 {name: DataFrame(prob_stressed, state_filtered)}。
    プール放出・遷移の推定は tilt 非依存なので1回だけ行い、filtered だけ tilt 別に評価する
    （モデル不変・数値は単体版と同一＝回帰テストで保証）。
    """
    target = feat_panel[target_code]
    idx = target.index
    Xt = target.to_numpy(dtype=float)
    n, K = len(Xt), n_states
    probs = {name: np.full((n, K), np.nan) for name in tilts}
    raw = {name: np.full(n, -1, dtype=int) for name in tilts}
    model: Optional[StudentTHMM] = None

    for t in range(warmup, n):
        lo = 0 if window is None else max(0, t + 1 - window)
        win = _standardize_past(Xt[lo:t + 1])
        if len(win) < warmup or not np.isfinite(win).all():
            continue
        if model is None or (t - warmup) % refit_every == 0:
            _, peers = _peers_asof(membership, target_code, idx[t])
            stacks = []
            for c in peers[:max_pool_codes]:
                if c not in feat_panel:
                    continue
                w = feat_panel[c].loc[:idx[t]]
                w = w.to_numpy(dtype=float)[-window:] if window else w.to_numpy(dtype=float)
                w = _standardize_past(w)
                if len(w) >= warmup and np.isfinite(w).all():
                    stacks.append(w)
            if len(stacks) >= 2:
                means, covars = fit_pooled_emissions(
                    np.vstack(stacks), n_states=K, dof=dof, n_init=n_init,
                    random_state=random_state, align_col=align_col)
                model = StudentTHMM(n_states=K, dof=dof, sticky_kappa=sticky_kappa, random_state=random_state)
                model.means_, model.covars_ = means, covars
                model.startprob_ = np.full(K, 1.0 / K)
                model.transmat_ = np.full((K, K), 0.1 / max(K - 1, 1))
                np.fill_diagonal(model.transmat_, 0.9)
                model.fit_transmat_only(win)
            else:
                model = fit_best(win, n_states=K, dof=dof, sticky_kappa=sticky_kappa,
                                 n_init=n_init, random_state=random_state)
                model.align_by(model.means_[:, align_col])
        for name, tilt in tilts.items():
            f = (model.filtered_proba(win)[-1] if tilt is None
                 else model.filtered_proba_injected(win, tilt[lo:t + 1])[-1])
            probs[name][t] = f
            raw[name][t] = int(np.argmax(f))

    out = {}
    for name in tilts:
        valid = raw[name] >= 0
        sd = raw[name].copy()
        if valid.any():
            sd[valid] = _min_dwell_filter(raw[name][valid], min_dwell)
        df = pd.DataFrame(index=idx)
        df["state_filtered"] = np.where(valid, sd, np.nan)
        df["prob_stressed"] = probs[name][:, K - 1]
        out[name] = df
    return out


# ============================================================ 比較指標
def stability_metrics(out: pd.DataFrame) -> dict[str, float]:
    """filtered 出力の安定性（ちらつき・持続・確率ジャンプ）を要約。"""
    s = out["state_filtered"].dropna()
    p = out["prob_stressed"].dropna()
    if len(s) < 3:
        return {"n": int(len(s)), "flip_rate": np.nan, "mean_dwell": np.nan, "prob_jump": np.nan}
    flips = float((s.values[1:] != s.values[:-1]).mean())
    runs, cur, L = [], s.iloc[0], 1
    for v in s.values[1:]:
        if v == cur:
            L += 1
        else:
            runs.append(L); cur, L = v, 1
    runs.append(L)
    return {
        "n": int(len(s)),
        "flip_rate": flips,                                   # 日次状態切替率（低いほど安定）
        "mean_dwell": float(np.mean(runs)),                   # 平均滞在日数（高いほど安定）
        "prob_jump": float(np.abs(np.diff(p.values)).mean()), # filtered 確率の日次ジャンプ（低=安定）
    }
