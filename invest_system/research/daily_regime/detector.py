"""daily_regime: 日足ボラ・流動性 多変量 Student-t HMM（per-stock・②）。

設計（接続設計 §5・設計書 §7）：
  - 放出は **Student-t**（固定 dof）。ファットテール・決算ジャンプを偽状態に吸わせない。
  - **sticky**（自己遷移へ Dirichlet 擬似カウント）＋ **min-dwell**（filtered 報告状態の
    オンライン・ヒステリシス）で日替わりちらつきを抑える。
  - 状態数は config 可変（2 決め打ちしない。2→3 は事前登録仮説で別途評価）。
  - 整列は `_align_states`（ボラ/Amihud 昇順 → calm…stressed）。
  - walk-forward：≤t のみで refit、終端 filtered（forward 正規化）を採用＝オンライン。

per-stock は pooling（③ pooling.py）のアブレーション対照として残す（データ枯渇で不安定でも
それが pooling の動機）。本モジュールは検知器コア＋per-stock walk-forward を提供する。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from invest_system.research.sector_regime.detectors import _align_states


# ============================================================ 放出（Student-t）
def _chol_mahal_logdet(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, reg: float):
    """マハラノビス距離 δ_t と log|Σ| を Cholesky で安定計算。"""
    d = X.shape[1]
    L = np.linalg.cholesky(cov + reg * np.eye(d))
    diff = (X - mean).T                                  # (d, T)
    sol = np.linalg.solve(L, diff)                       # L y = diff
    delta = np.sum(sol * sol, axis=0)                    # (T,)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return delta, log_det


def _log_mvt(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, dof: float, reg: float):
    """多変量 Student-t の対数密度（固定 dof）。"""
    d = X.shape[1]
    delta, log_det = _chol_mahal_logdet(X, mean, cov, reg)
    norm = (gammaln((dof + d) / 2.0) - gammaln(dof / 2.0)
            - 0.5 * (d * np.log(dof * np.pi) + log_det))
    return norm - 0.5 * (dof + d) * np.log1p(delta / dof)


# ============================================================ Student-t HMM
class StudentTHMM:
    """多変量 Student-t 放出・sticky 遷移の HMM（Baum-Welch / ECM）。"""

    def __init__(self, n_states: int = 2, dof: float = 4.0, sticky_kappa: float = 10.0,
                 n_iter: int = 80, tol: float = 1e-3, reg: float = 1e-6, random_state: int = 0):
        self.n_states = n_states
        self.dof = dof
        self.sticky_kappa = sticky_kappa
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg
        self.random_state = random_state

    # ---- 初期化（k-means） ----
    def _init(self, X: np.ndarray, rng) -> None:
        from sklearn.cluster import KMeans
        K, (T, d) = self.n_states, X.shape
        km = KMeans(n_clusters=K, n_init=3, random_state=int(rng.integers(0, 1 << 31))).fit(X)
        lbl = km.labels_
        self.means_ = np.array([X[lbl == k].mean(0) if (lbl == k).any() else X.mean(0)
                                for k in range(K)])
        self.covars_ = np.array([np.cov(X[lbl == k].T) + self.reg * np.eye(d)
                                 if (lbl == k).sum() > d else np.cov(X.T) + self.reg * np.eye(d)
                                 for k in range(K)]).reshape(K, d, d)
        self.startprob_ = np.full(K, 1.0 / K)
        self.transmat_ = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(self.transmat_, 0.9)

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([_log_mvt(X, self.means_[k], self.covars_[k], self.dof, self.reg)
                                for k in range(self.n_states)])

    def _forward_backward(self, log_e: np.ndarray):
        T, K = log_e.shape
        log_T = np.log(self.transmat_ + 1e-300)
        la = np.empty((T, K))
        la[0] = np.log(self.startprob_ + 1e-300) + log_e[0]
        for t in range(1, T):
            la[t] = logsumexp(la[t - 1][:, None] + log_T, axis=0) + log_e[t]
        loglik = logsumexp(la[-1])
        lb = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(log_T + (log_e[t + 1] + lb[t + 1])[None, :], axis=1)
        gamma = np.exp(la + lb - loglik)
        xi = np.zeros((K, K))
        for t in range(T - 1):
            xi += np.exp(la[t][:, None] + log_T + (log_e[t + 1] + lb[t + 1])[None, :] - loglik)
        return loglik, gamma, xi

    def _update_emissions(self, X: np.ndarray, gamma: np.ndarray) -> None:
        T, d = X.shape
        for k in range(self.n_states):
            delta, _ = _chol_mahal_logdet(X, self.means_[k], self.covars_[k], self.reg)
            u = (self.dof + d) / (self.dof + delta)            # t の EM 重み
            gw = gamma[:, k] * u
            mu = (gw[:, None] * X).sum(0) / (gw.sum() + 1e-300)
            diff = X - mu
            cov = np.einsum('t,ti,tj->ij', gamma[:, k] * u, diff, diff) / (gamma[:, k].sum() + 1e-300)
            self.means_[k] = mu
            self.covars_[k] = cov + self.reg * np.eye(d)

    def fit(self, X: np.ndarray) -> "StudentTHMM":
        rng = np.random.default_rng(self.random_state)
        self._init(X, rng)
        prev = -np.inf
        for it in range(self.n_iter):
            loglik, gamma, xi = self._forward_backward(self._log_emission(X))
            self.startprob_ = (gamma[0] + 1e-8) / (gamma[0].sum() + self.n_states * 1e-8)
            num = xi + self.sticky_kappa * np.eye(self.n_states)   # sticky Dirichlet 擬似カウント
            self.transmat_ = num / num.sum(1, keepdims=True)
            self._update_emissions(X, gamma)
            if it > 0 and abs(loglik - prev) < self.tol:
                break
            prev = loglik
        self.loglik_ = float(loglik)
        return self

    def fit_transmat_only(self, X: np.ndarray, n_iter: int = 40) -> "StudentTHMM":
        """放出（means/covars）を**固定**し、startprob/transmat のみ EM 更新。

        pooling（③）で放出をプール推定値に固定し、銘柄別の遷移ダイナミクスのみ当てる用途。
        means_/covars_/startprob_/transmat_ を事前設定してから呼ぶ。
        """
        prev = -np.inf
        for it in range(n_iter):
            loglik, gamma, xi = self._forward_backward(self._log_emission(X))
            self.startprob_ = (gamma[0] + 1e-8) / (gamma[0].sum() + self.n_states * 1e-8)
            num = xi + self.sticky_kappa * np.eye(self.n_states)
            self.transmat_ = num / num.sum(1, keepdims=True)
            if it > 0 and abs(loglik - prev) < self.tol:
                break
            prev = loglik
        self.loglik_ = float(loglik)
        return self

    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        """オンライン filtered 事後 P(state_t | x_{≤t})（forward 正規化）。"""
        log_e = self._log_emission(X)
        T, K = log_e.shape
        log_T = np.log(self.transmat_ + 1e-300)
        out = np.empty((T, K))
        la = np.log(self.startprob_ + 1e-300) + log_e[0]
        out[0] = np.exp(la - logsumexp(la))
        for t in range(1, T):
            pred = logsumexp(np.log(out[t - 1] + 1e-300)[:, None] + log_T, axis=0)
            a = pred + log_e[t]
            out[t] = np.exp(a - logsumexp(a))
        return out

    def filtered_proba_injected(self, X: np.ndarray, log_tilt: np.ndarray) -> np.ndarray:
        """方式A（事前分布注入）：各ステップの状態**事前**に log_tilt を加えた filtered。

        log_tilt: (T, K)。週足アンカー由来（bear→stressed 事前↑）。log_tilt=0 で
        filtered_proba と一致（注入なし＝ベースライン）。オンライン（forward のみ・≤t）。
        """
        log_e = self._log_emission(X)
        T, K = log_e.shape
        log_T = np.log(self.transmat_ + 1e-300)
        out = np.empty((T, K))
        la = np.log(self.startprob_ + 1e-300) + log_tilt[0] + log_e[0]
        out[0] = np.exp(la - logsumexp(la))
        for t in range(1, T):
            pred = logsumexp(np.log(out[t - 1] + 1e-300)[:, None] + log_T, axis=0)
            a = pred + log_tilt[t] + log_e[t]              # 事前（pred）に注入
            out[t] = np.exp(a - logsumexp(a))
        return out

    def align_by(self, key_per_state: np.ndarray) -> "StudentTHMM":
        """状態を key 昇順（calm→stressed）に並べ替えてラベルの意味を固定。"""
        order = _align_states(np.arange(self.n_states), np.asarray(key_per_state), ascending=True)
        perm = np.array(order)
        self.means_ = self.means_[perm]
        self.covars_ = self.covars_[perm]
        self.startprob_ = self.startprob_[perm]
        self.transmat_ = self.transmat_[np.ix_(perm, perm)]
        self._order = perm
        return self


def fit_best(X: np.ndarray, *, n_states: int, dof: float, sticky_kappa: float,
             n_init: int, random_state: int, reg: float = 1e-6) -> StudentTHMM:
    """複数初期値で再起動し最良尤度の Student-t HMM を返す（EM 初期値依存対策）。"""
    best, best_ll = None, -np.inf
    for j in range(max(1, n_init)):
        m = StudentTHMM(n_states=n_states, dof=dof, sticky_kappa=sticky_kappa,
                        reg=reg, random_state=random_state + 1000 * j).fit(X)
        if np.isfinite(m.loglik_) and m.loglik_ > best_ll:
            best, best_ll = m, m.loglik_
    if best is None:                                       # 退避（全滅時）
        best = StudentTHMM(n_states=n_states, dof=dof, sticky_kappa=sticky_kappa,
                           reg=reg, random_state=random_state).fit(X)
    return best


def fit_t_mixture(X: np.ndarray, *, n_states: int, dof: float, n_init: int,
                  random_state: int, reg: float = 1e-6, n_iter: int = 120, tol: float = 1e-4):
    """Student-t 混合（遷移なし）の EM。プール放出推定用（FB 不要・ベクトル化で高速）。

    スタックしたティア横断観測から calm/stressed クラスタの位置のみを取り出す。時間連続でない
    プールに HMM 遷移を当てるのは不適切なので混合で放出のみ推定する。返り値 (means, covars, weights)。
    """
    from sklearn.cluster import KMeans
    T, d = X.shape
    best, best_ll = None, -np.inf
    for j in range(max(1, n_init)):
        rng = np.random.default_rng(random_state + 100 * j)
        lbl = KMeans(n_clusters=n_states, n_init=2,
                     random_state=int(rng.integers(0, 1 << 31))).fit(X).labels_
        means = np.array([X[lbl == k].mean(0) if (lbl == k).any() else X.mean(0)
                          for k in range(n_states)])
        covars = np.array([np.cov(X[lbl == k].T) + reg * np.eye(d) if (lbl == k).sum() > d
                           else np.cov(X.T) + reg * np.eye(d) for k in range(n_states)]).reshape(n_states, d, d)
        weights = np.array([max((lbl == k).mean(), 1e-3) for k in range(n_states)])
        weights /= weights.sum()
        prev = -np.inf
        for it in range(n_iter):
            log_e = np.column_stack([_log_mvt(X, means[k], covars[k], dof, reg) for k in range(n_states)])
            log_post = log_e + np.log(weights + 1e-300)
            ll_row = logsumexp(log_post, axis=1)
            loglik = float(ll_row.sum())
            gamma = np.exp(log_post - ll_row[:, None])
            for k in range(n_states):
                delta, _ = _chol_mahal_logdet(X, means[k], covars[k], reg)
                u = (dof + d) / (dof + delta)
                gw = gamma[:, k] * u
                mu = (gw[:, None] * X).sum(0) / (gw.sum() + 1e-300)
                diff = X - mu
                covars[k] = (np.einsum('t,ti,tj->ij', gamma[:, k] * u, diff, diff)
                             / (gamma[:, k].sum() + 1e-300)) + reg * np.eye(d)
                means[k] = mu
            weights = gamma.mean(0) + 1e-8
            weights /= weights.sum()
            if it > 0 and abs(loglik - prev) < tol * (abs(prev) + 1e-9):
                break
            prev = loglik
        if np.isfinite(loglik) and loglik > best_ll:
            best, best_ll = (means.copy(), covars.copy(), weights.copy()), loglik
    return best


def _min_dwell_filter(states: np.ndarray, min_dwell: int) -> np.ndarray:
    """min-dwell オンライン・ヒステリシス：新状態が min_dwell 日連続で優勢になるまで切替えない。"""
    if min_dwell <= 1 or len(states) == 0:
        return states
    out = states.copy()
    cur = states[0]
    cand, run = cur, 0
    for t in range(len(states)):
        if states[t] == cur:
            cand, run = cur, 0
        elif states[t] == cand:
            run += 1
            if run >= min_dwell:
                cur, run = cand, 0
        else:
            cand, run = states[t], 1
        out[t] = cur
    return out


def _standardize_past(win: np.ndarray):
    mu = np.nanmean(win, axis=0)
    sd = np.nanstd(win, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (win - mu) / sd


def walk_forward_t_hmm(
    features: pd.DataFrame,
    *,
    n_states: int = 2,
    dof: float = 4.0,
    sticky_kappa: float = 10.0,
    min_dwell: int = 3,
    refit_every: int = 20,
    window: Optional[int] = 756,
    warmup: int = 252,
    align_col: int = 0,
    n_init: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """per-stock 日足ボラ流動性 HMM の walk-forward（終端 filtered・オンライン）。

    Returns: DataFrame(index=日付) — state_filtered（min-dwell 後）, prob_stressed,
             p_state{0..K-1}, refit（その日に再学習したか）。
    """
    idx = features.index
    X_all = features.to_numpy(dtype=float)
    n, d = X_all.shape
    K = n_states
    prob = np.full((n, K), np.nan)
    raw_state = np.full(n, -1, dtype=int)
    refit_flag = np.zeros(n, dtype=bool)
    model: Optional[StudentTHMM] = None

    for t in range(warmup, n):
        lo = 0 if window is None else max(0, t + 1 - window)
        win = _standardize_past(X_all[lo:t + 1])
        if len(win) < warmup or not np.isfinite(win).all():
            continue
        if model is None or (t - warmup) % refit_every == 0:
            model = fit_best(win, n_states=K, dof=dof, sticky_kappa=sticky_kappa,
                             n_init=n_init, random_state=random_state)
            key = model.means_[:, align_col]               # ボラ/Amihud 列で整列
            model.align_by(key)
            refit_flag[t] = True
        f = model.filtered_proba(win)[-1]                  # 終端 filtered = オンライン
        prob[t] = f
        raw_state[t] = int(np.argmax(f))

    valid = raw_state >= 0
    state_dwell = raw_state.copy()
    if valid.any():
        sv = _min_dwell_filter(raw_state[valid], min_dwell)
        state_dwell[valid] = sv

    out = pd.DataFrame(index=idx)
    out["state_filtered"] = np.where(valid, state_dwell, np.nan)
    out["prob_stressed"] = prob[:, K - 1]                  # 整列後の最終状態＝最高ボラ/Amihud
    for k in range(K):
        out[f"p_state{k}"] = prob[:, k]
    out["refit"] = refit_flag
    return out
