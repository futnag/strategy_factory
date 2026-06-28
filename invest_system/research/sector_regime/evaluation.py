"""レジーム検知の多角的評価と複合スコアリング。"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import SCORE_WEIGHTS
from .detectors import RegimeOutput


def _align_eval_frame(
    output: RegimeOutput,
    weekly: pd.DataFrame,
    sector: str,
) -> pd.DataFrame:
    """レジーム出力と翌週ターゲットを整列。"""
    sub = weekly[weekly["sector"] == sector].set_index("date").sort_index()
    df = pd.DataFrame(index=sub.index)
    df["regime"] = output.regime
    df["regime_prob"] = output.regime_prob
    df["log_return_w"] = sub["log_return_w"]
    df["realized_vol_w"] = sub["realized_vol_w"]
    df["next_log_return_w"] = sub["log_return_w"].shift(-1)
    df["next_realized_vol_w"] = sub["realized_vol_w"].shift(-1)
    df["next_return_sign"] = np.sign(df["next_log_return_w"])
    return df.dropna(subset=["regime", "next_log_return_w", "next_realized_vol_w"])


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return np.nan
    if a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return 0.0
    return float(a[m].corr(b[m]))


def _r2_improvement(y: pd.Series, x: pd.Series) -> float:
    """x を加えたときの R² 改善度（単純 OLS）。"""
    m = y.notna() & x.notna()
    if m.sum() < 20:
        return np.nan
    yy = y[m].values
    xx = x[m].values
    # ベースライン: 定数のみ
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    # モデル: y ~ x
    slope, intercept, _, _, _ = stats.linregress(xx, yy)
    pred = intercept + slope * xx
    ss_res = np.sum((yy - pred) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, r2))


def score_prediction(df: pd.DataFrame) -> dict[str, float]:
    """予測力スコア（相関・R²・方向的中率）。"""
    corr_vol = _safe_corr(df["regime_prob"], df["next_realized_vol_w"])
    corr_ret = _safe_corr(df["regime"], df["next_log_return_w"])
    r2_vol = _r2_improvement(df["next_realized_vol_w"], df["regime_prob"])
    r2_ret = _r2_improvement(df["next_log_return_w"], df["regime"].astype(float))

    # 方向性: 高ボラレジーム時にリターン符号を予測
    high = df["regime_prob"] > df["regime_prob"].median()
    pred_sign = np.where(high, -1.0, 1.0)
    hit = (np.sign(pred_sign) == df["next_return_sign"]).mean()

    # 0-1 正規化（経験的レンジ）
    s_corr = np.nanmean([abs(corr_vol), abs(corr_ret)])
    s_corr = min(1.0, s_corr / 0.3) if np.isfinite(s_corr) else 0.0
    s_r2 = np.nanmean([r2_vol, r2_ret])
    s_r2 = min(1.0, s_r2 / 0.1) if np.isfinite(s_r2) else 0.0
    s_hit = float(hit) if np.isfinite(hit) else 0.5

    composite = 0.4 * s_corr + 0.35 * s_r2 + 0.25 * s_hit
    return {
        "pred_corr_vol": corr_vol,
        "pred_corr_ret": corr_ret,
        "pred_r2_vol": r2_vol,
        "pred_r2_ret": r2_ret,
        "pred_hit_rate": hit,
        "pred_score": composite,
    }


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """ガウス分布間の KL divergence（対称化）。"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if len(p) < 5 or len(q) < 5:
        return 0.0
    mp, sp = p.mean(), p.std()
    mq, sq = q.mean(), q.std()
    if sp < 1e-12 or sq < 1e-12:
        return 0.0
    kl_pq = np.log(sq / sp) + (sp ** 2 + (mp - mq) ** 2) / (2 * sq ** 2) - 0.5
    kl_qp = np.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2 * sp ** 2) - 0.5
    return float(0.5 * (kl_pq + kl_qp))


def score_regime_quality(df: pd.DataFrame) -> dict[str, float]:
    """レジームの質（分離度・持続期間・切替頻度）。"""
    regimes = df["regime"].astype(int)
    uniq = regimes.unique()
    if len(uniq) < 2:
        return {
            "sep_mean_diff": 0.0, "sep_kl": 0.0,
            "avg_duration": 0.0, "switch_freq": 0.0,
            "quality_score": 0.0,
        }

    # 分離度: 翌週ボラの状態間差
    vol_by_state = {
        int(s): df.loc[regimes == s, "next_realized_vol_w"].dropna().values
        for s in uniq
    }
    pairs = list(uniq)
    mean_diffs, kls = [], []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = vol_by_state[int(pairs[i])], vol_by_state[int(pairs[j])]
            if len(a) > 5 and len(b) > 5:
                mean_diffs.append(abs(a.mean() - b.mean()))
                kls.append(_kl_divergence(a, b))

    # 持続期間
    changes = (regimes != regimes.shift(1)).cumsum()
    durations = regimes.groupby(changes).size()
    avg_dur = float(durations.mean())
    switch_freq = float((regimes != regimes.shift(1)).sum() / len(regimes))

    sep_md = float(np.mean(mean_diffs)) if mean_diffs else 0.0
    sep_kl = float(np.mean(kls)) if kls else 0.0

    s_sep = min(1.0, sep_md / 0.05) * 0.5 + min(1.0, sep_kl / 0.5) * 0.5
    s_dur = min(1.0, avg_dur / 20.0)
    s_freq = max(0.0, 1.0 - switch_freq * 10.0)  # 過度な切替はペナルティ

    composite = 0.5 * s_sep + 0.3 * s_dur + 0.2 * s_freq
    return {
        "sep_mean_diff": sep_md,
        "sep_kl": sep_kl,
        "avg_duration": avg_dur,
        "switch_freq": switch_freq,
        "quality_score": composite,
    }


def score_stability(
    output: RegimeOutput,
    features: pd.DataFrame,
    *,
    n_boot: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """安定性: サブサンプル間のラベル一致率（初期値依存性の低さの代理）。"""
    from .detectors import walk_forward_hmm

    rng = np.random.default_rng(random_state)
    base = output.regime.dropna()
    if len(base) < 50:
        return {"stability_agreement": 0.5, "stability_score": 0.5}

    agreements = []
    n = len(features)
    for _ in range(n_boot):
        cut = int(rng.integers(int(n * 0.7), n))
        sub_feat = features.iloc[:cut]
        alt = walk_forward_hmm(
            sub_feat,
            sector=output.sector,
            n_states=output.n_states,
            random_state=int(rng.integers(0, 10000)),
        )
        common = base.index.intersection(alt.regime.dropna().index)
        if len(common) < 30:
            continue
        a = base.loc[common].astype(int)
        b = alt.regime.loc[common].astype(int)
        # ラベルは順序不定なので最良マッチング（2状態は反転許容）
        agree = (a == b).mean()
        agree_flip = (a == (1 - b)).mean()
        agreements.append(max(agree, agree_flip))

    agr = float(np.mean(agreements)) if agreements else 0.5
    return {"stability_agreement": agr, "stability_score": agr}


def score_economic(df: pd.DataFrame) -> dict[str, float]:
    """経済的意味: レジーム間のリターン・ボラの t 検定。"""
    regimes = df["regime"].astype(int)
    uniq = regimes.unique()
    if len(uniq) < 2:
        return {"econ_t_ret": np.nan, "econ_t_vol": np.nan, "econ_score": 0.0}

    a = uniq[0]
    b = uniq[1]
    ret_a = df.loc[regimes == a, "next_log_return_w"].dropna()
    ret_b = df.loc[regimes == b, "next_log_return_w"].dropna()
    vol_a = df.loc[regimes == a, "next_realized_vol_w"].dropna()
    vol_b = df.loc[regimes == b, "next_realized_vol_w"].dropna()

    t_ret, p_ret = (np.nan, 1.0)
    t_vol, p_vol = (np.nan, 1.0)
    if len(ret_a) > 10 and len(ret_b) > 10:
        t_ret, p_ret = stats.ttest_ind(ret_a, ret_b, equal_var=False)
    if len(vol_a) > 10 and len(vol_b) > 10:
        t_vol, p_vol = stats.ttest_ind(vol_a, vol_b, equal_var=False)

    sig_ret = 1.0 - min(1.0, p_ret) if np.isfinite(p_ret) else 0.0
    sig_vol = 1.0 - min(1.0, p_vol) if np.isfinite(p_vol) else 0.0
    composite = 0.5 * sig_ret + 0.5 * sig_vol

    return {
        "econ_t_ret": float(t_ret) if np.isfinite(t_ret) else np.nan,
        "econ_p_ret": float(p_ret) if np.isfinite(p_ret) else np.nan,
        "econ_t_vol": float(t_vol) if np.isfinite(t_vol) else np.nan,
        "econ_p_vol": float(p_vol) if np.isfinite(p_vol) else np.nan,
        "econ_score": composite,
    }


def composite_score(metrics: dict[str, float]) -> float:
    """重み付き複合スコア（0-1）。"""
    return float(
        SCORE_WEIGHTS["prediction"] * metrics.get("pred_score", 0.0)
        + SCORE_WEIGHTS["regime_quality"] * metrics.get("quality_score", 0.0)
        + SCORE_WEIGHTS["stability"] * metrics.get("stability_score", 0.0)
        + SCORE_WEIGHTS["economic"] * metrics.get("econ_score", 0.0)
    )


def evaluate_method(
    output: RegimeOutput,
    weekly: pd.DataFrame,
    features: pd.DataFrame,
    *,
    skip_stability: bool = False,
) -> dict[str, float]:
    """1手法の全評価指標を算出。"""
    df = _align_eval_frame(output, weekly, output.sector)
    if df.empty:
        return {"composite_score": 0.0, "n_eval": 0}

    metrics: dict[str, float] = {"n_eval": float(len(df))}
    metrics.update(score_prediction(df))
    metrics.update(score_regime_quality(df))
    metrics.update(score_economic(df))
    if not skip_stability and output.method.startswith("hmm"):
        metrics.update(score_stability(output, features))
    else:
        metrics.update({"stability_agreement": 0.5, "stability_score": 0.5})
    metrics["composite_score"] = composite_score(metrics)
    return metrics


def rank_methods(
    results: list[tuple[RegimeOutput, dict[str, float]]],
) -> pd.DataFrame:
    """手法ランキング表。"""
    rows = []
    for out, met in results:
        rows.append({
            "sector": out.sector,
            "method": out.method,
            "n_states": out.n_states,
            "composite_score": met.get("composite_score", 0.0),
            "pred_score": met.get("pred_score", 0.0),
            "quality_score": met.get("quality_score", 0.0),
            "stability_score": met.get("stability_score", 0.0),
            "econ_score": met.get("econ_score", 0.0),
            "pred_hit_rate": met.get("pred_hit_rate", np.nan),
            "avg_duration": met.get("avg_duration", np.nan),
            "switch_freq": met.get("switch_freq", np.nan),
            "n_eval": met.get("n_eval", 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["sector", "composite_score"], ascending=[True, False])


def best_per_sector(ranking: pd.DataFrame) -> pd.DataFrame:
    """セクターごとのベスト手法。"""
    if ranking.empty:
        return ranking
    return (
        ranking.sort_values("composite_score", ascending=False)
        .groupby("sector", as_index=False)
        .first()
    )


def selection_rationale(row: pd.Series) -> str:
    """ベスト手法の選定理由テキスト。"""
    parts = [
        f"複合スコア {row['composite_score']:.3f} で最高。",
        f"予測力={row['pred_score']:.3f}（Hit率 {row.get('pred_hit_rate', np.nan):.1%}）、",
        f"レジーム質={row['quality_score']:.3f}（平均持続 {row.get('avg_duration', np.nan):.1f}週）、",
        f"安定性={row['stability_score']:.3f}、経済的意味={row['econ_score']:.3f}。",
    ]
    return "".join(parts)