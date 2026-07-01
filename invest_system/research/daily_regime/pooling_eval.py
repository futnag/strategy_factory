"""daily_regime: pooling 良否の**判定可能な**評価（#1 放出安定性＋#3 推定器統制）。

T2 比較が判定不能だった根本原因2つを断つ：
  #1 持続性機構（sticky/min-dwell）が放出の不安定を隠す → filtered 出力でなく**放出パラメータ**
     （stressed クラスタ平均・calm–stressed gap）の **refit 間横断 CV** で測る（持続性機構の*前*）。
  #3 pooled=t混合 / per-stock=HMM-FB の土俵差 → **両者を同一推定器（t 混合）に揃える**。
     per-stock 放出も t 混合で推定し、唯一の差を「自銘柄窓 vs ティア・プール窓」に限定する。

判定基準は**事前登録**（regime_detection_pooling_preregistration.md・本モジュール定数）。事後に
動かさない。λ（pooling 強度）は固定・非最適化（CV を下げにλを動かさない＝二重過剰適合の回避）。

PIT：プール窓は as-of r のティア（≤r）の obs バッグ。t 混合は順序非依存で時間越え推定をしない
（多系列 HMM の refit 境界リークを作らない）。(b) 多系列 HMM 化は後置（その際 PIT を別途検証）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from invest_system.research.sector_regime.detectors import _align_states

from .contracts import LookAheadError
from .detector import _log_mvt, fit_t_mixture

# === 事前登録した判定閾値（事後に動かさない） ===
GAP_CV_RATIO = 0.8         # pooled gap-CV ≤ 0.8 × per-stock（≥20% 低減）で「有意に安定」
MAJORITY = 2.0 / 3.0       # ゾーン内ターゲットの ≥2/3 で pooled 改善なら採用
MIN_WEIGHT_FLOOR = 0.05    # 状態 weight < 0.05 ＝ collapse（退化）
COND_CEILING = 1.0e4       # 共分散 条件数 > 1e4 ＝ near-singular（退化）


def emission_diagnostics(means: np.ndarray, covars: np.ndarray, weights: np.ndarray,
                         align_col: int) -> dict:
    """整列後の放出から診断量を抽出（stressed 平均・gap・最小weight・最大条件数）。"""
    order = _align_states(np.arange(len(weights)), means[:, align_col], ascending=True)
    calm, stressed = order[0], order[-1]
    return {
        "stressed_mean": float(means[stressed, align_col]),
        "gap": float(means[stressed, align_col] - means[calm, align_col]),
        "min_weight": float(np.min(weights)),
        "max_cond": float(max(np.linalg.cond(c) for c in covars)),
    }


def fit_emissions(X: np.ndarray, *, n_states: int, dof: float, n_init: int,
                  random_state: int, align_col: int):
    """同一推定器（t 混合）で放出を推定し診断量を返す（#3 統制）。"""
    res = fit_t_mixture(X, n_states=n_states, dof=dof, n_init=n_init, random_state=random_state)
    if res is None:
        return None
    means, covars, weights = res
    return emission_diagnostics(means, covars, weights, align_col)


def track_emissions(window_iter, *, n_states: int, dof: float, n_init: int,
                    random_state: int, align_col: int) -> pd.DataFrame:
    """refit ごとの標準化済み窓 (refit_date, X_std) から放出診断を時系列で収集。"""
    recs = []
    for r, X in window_iter:
        if X is None or len(X) < n_states + 2 or not np.isfinite(X).all():
            continue
        d = fit_emissions(X, n_states=n_states, dof=dof, n_init=n_init,
                          random_state=random_state, align_col=align_col)
        if d is not None:
            d["refit"] = r
            recs.append(d)
    return pd.DataFrame(recs)


def stability_summary(track: pd.DataFrame) -> dict:
    """放出トラックの安定性要約：gap の横断 CV と退化率（持続性機構の*前*）。"""
    if track is None or len(track) < 3:
        return {"n_refits": 0 if track is None else int(len(track)),
                "gap_cv": np.nan, "stressed_cv": np.nan,
                "degeneracy_rate": np.nan, "mean_min_weight": np.nan}
    gap = track["gap"].abs()
    sm = track["stressed_mean"]
    degen = ((track["min_weight"] < MIN_WEIGHT_FLOOR) | (track["max_cond"] > COND_CEILING)).mean()
    return {
        "n_refits": int(len(track)),
        "gap_cv": float(gap.std() / (gap.mean() + 1e-12)),
        "stressed_cv": float(sm.std() / (abs(sm.mean()) + 1e-12)),
        "degeneracy_rate": float(degen),
        "mean_min_weight": float(track["min_weight"].mean()),
    }


def verdict(per_stock: dict[str, dict], pooled: dict[str, dict]) -> dict:
    """事前登録基準でゾーンの pooling 採否を判定（事後に基準を動かさない）。

    採用条件：ターゲットの ≥MAJORITY で pooled が gap-CV を GAP_CV_RATIO 倍以下に下げ、かつ
    退化率を悪化させない。
    """
    better, total = 0, 0
    for code, ps in per_stock.items():
        pl = pooled.get(code)
        if pl is None or not np.isfinite(ps.get("gap_cv", np.nan)) or not np.isfinite(pl.get("gap_cv", np.nan)):
            continue
        total += 1
        if pl["gap_cv"] <= GAP_CV_RATIO * ps["gap_cv"] and pl["degeneracy_rate"] <= ps["degeneracy_rate"]:
            better += 1
    frac = better / total if total else 0.0
    return {"n": total, "n_better": better, "frac_better": frac,
            "adopt_pooling": bool(total > 0 and frac >= MAJORITY)}


# === 保留予測 LL（分散＋バイアスを同時に含む正味の裁定指標） ===
def mixture_loglik(means: np.ndarray, covars: np.ndarray, weights: np.ndarray,
                   X: np.ndarray, dof: float, reg: float = 1e-6) -> np.ndarray:
    """Student-t 混合の各観測の対数尤度（窓外スコアリング用）。"""
    log_e = np.column_stack([_log_mvt(X, means[k], covars[k], dof, reg) for k in range(len(weights))])
    return logsumexp(log_e + np.log(np.asarray(weights) + 1e-300), axis=1)


def assert_future_block(fit_asof, heldout_index) -> None:
    """held-out 区間が fit カットオフより**厳密に後**であることを実行時アサート（釘③・PIT）。"""
    fit_asof = pd.Timestamp(fit_asof)
    idx = pd.DatetimeIndex(heldout_index)
    if len(idx) and (idx <= fit_asof).any():
        raise LookAheadError(
            f"held-out に fit_asof {fit_asof} 以前の点 {idx[idx <= fit_asof][0]}（PIT 違反）"
        )


def _zwin(feat: pd.DataFrame, r, window: int):
    """≤r の末尾 window を、その窓の平均/分散で z 化（PIT）。(z, mu, sd) を返す。"""
    w = feat.loc[:r].tail(window)
    if len(w) < 5:
        return None, None, None
    mu = w.mean()
    sd = w.std().replace(0.0, 1.0)
    z = ((w - mu) / sd).to_numpy(float)
    if not np.isfinite(z).all():
        return None, None, None
    return z, mu, sd


def pooled_emissions_by_refit(peer_feats: list[pd.DataFrame], refit_dates, *,
                              window: int, n_states: int, dof: float,
                              n_init: int, random_state: int) -> dict:
    """各 refit でティア・プール（peer の自己 z 窓スタック）から t 混合放出を推定（≤r・PIT）。"""
    out = {}
    for r in refit_dates:
        stk = []
        for f in peer_feats:
            z, _, _ = _zwin(f, r, window)
            if z is not None and len(z) >= n_states + 2:
                stk.append(z)
        if len(stk) >= 2:
            res = fit_t_mixture(np.vstack(stk), n_states=n_states, dof=dof,
                                n_init=n_init, random_state=random_state)
            if res is not None:
                out[r] = res
    return out


def heldout_ll_for_target(target_feat: pd.DataFrame, pooled_by_refit: dict, refit_dates, *,
                          window: int, horizon: int, n_states: int, dof: float,
                          n_init: int, random_state: int):
    """各 refit で per-stock / pooled 放出を、target の**窓外未来 (r, r+horizon]** でスコア。

    per-stock z 空間（target は自己 ≤r 統計で z 化、held-out も同統計で z 化＝PIT）。
    返り値 (per_stock 平均LL, pooled 平均LL, 使用refit数)。
    """
    ps, pl = [], []
    for r in refit_dates:
        if r not in pooled_by_refit:
            continue
        z, mu, sd = _zwin(target_feat, r, window)
        if z is None or len(z) < n_states + 5:
            continue
        future = target_feat.loc[r:].iloc[1:1 + horizon]      # 厳密に r より後
        if len(future) < 3:
            continue
        assert_future_block(r, future.index)                  # 釘③：PIT 越え禁止
        ho = ((future - mu) / sd).to_numpy(float)             # ≤r 統計で z 化（リークなし）
        if not np.isfinite(ho).all():
            continue
        res_ps = fit_t_mixture(z, n_states=n_states, dof=dof, n_init=n_init, random_state=random_state)
        if res_ps is None:
            continue
        ps.append(float(mixture_loglik(*res_ps, ho, dof).mean()))
        pl.append(float(mixture_loglik(*pooled_by_refit[r], ho, dof).mean()))
    return (float(np.mean(ps)) if ps else np.nan,
            float(np.mean(pl)) if pl else np.nan, len(ps))
