"""daily_regime: 週足方向性アンカー（契約1の生成・業種指数・フォールバックB）。

⚠ step 0–1 範囲：方向レジームは**スタブ**（モメンタム符号のヒューリスティック）。本物の
   週足方向性HMM（filtered）は step 5（multiscale）で差し替える。本モジュールの目的は
   「契約スキーマ＋§1-4 可用日ラグ＋フォールバックBの切替フラグ」の配線確定であって、
   レジームの質ではない。

フォールバックB（接続設計 §4 確定）：当該週のセクター構成銘柄数 < N（固定・非最適化）の
   とき広域（等加重マーケット）代理の方向へ切替え、anchor_source='market' で**監査可能に記録**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .config import (
    ANCHOR_MOM_WEEKS,
    ANCHOR_NEUTRAL_BAND,
    FALLBACK_MIN_CONSTITUENTS,
    PUBLISH_LAG_BDAYS,
    WEEKLY_FREQ,
)
from .contracts import validate_anchor


def _sector_indices(adj_close: pd.DataFrame, s33: pd.Series):
    """S33 等加重日次リターン → セクター指数（close 代理）・構成銘柄数（PIT）・マーケット代理。"""
    ret = adj_close.pct_change(fill_method=None)
    sec = s33.reindex([str(c) for c in adj_close.columns])
    sec.index = adj_close.columns
    levels, counts = {}, {}
    for s, cols in sec.dropna().groupby(sec.dropna()):
        cset = list(cols.index)
        sret = ret[cset].mean(axis=1)
        levels[s] = (1.0 + sret.fillna(0.0)).cumprod() * 100.0
        counts[s] = ret[cset].notna().sum(axis=1)          # 当日取引銘柄数（≤t）
    mkt = (1.0 + ret.mean(axis=1).fillna(0.0)).cumprod() * 100.0
    return pd.DataFrame(levels), pd.DataFrame(counts), mkt


def _stub_probs(mom: pd.Series, band: float):
    """⚠スタブ：モメンタムから bull/bear/neutral の擬似 filtered 確率（softmax, Σ=1）。

    本HMMの filtered 確率の**形だけ**を満たす代理。step 5 で本物に差し替える。
    """
    band = max(band, 1e-9)
    s_bull = np.clip(np.maximum(mom, 0.0) / band, 0.0, 10.0)
    s_bear = np.clip(np.maximum(-mom, 0.0) / band, 0.0, 10.0)
    s_neu = pd.Series(1.0, index=mom.index)
    e_bull, e_bear, e_neu = np.exp(s_bull), np.exp(s_bear), np.exp(s_neu)
    z = e_bull + e_bear + e_neu
    return e_bear / z, e_neu / z, e_bull / z


def build_weekly_anchor(
    adj_close: pd.DataFrame,
    s33: pd.Series,
    *,
    freq: str = WEEKLY_FREQ,
    mom_weeks: int = ANCHOR_MOM_WEEKS,
    band: float = ANCHOR_NEUTRAL_BAND,
    min_constituents: int = FALLBACK_MIN_CONSTITUENTS,
    publish_lag: int = PUBLISH_LAG_BDAYS,
) -> pd.DataFrame:
    """業種指数（S33 代理）から WeeklyDirectionalAnchor を生成（スキーマ検証済みで返す）。"""
    levels, counts, mkt = _sector_indices(adj_close, s33)
    wk = levels.resample(freq).last()
    wk_cnt = counts.resample(freq).last()
    wk_mkt = mkt.resample(freq).last()
    mkt_mom = wk_mkt / wk_mkt.shift(mom_weeks) - 1.0

    rows = []
    for sector in wk.columns:
        lvl = wk[sector].dropna()
        if len(lvl) <= mom_weeks:
            continue
        mom = lvl / lvl.shift(mom_weeks) - 1.0
        cnt = wk_cnt[sector].reindex(lvl.index)
        use_market = cnt < min_constituents                 # フォールバックB
        mom_eff = mom.where(~use_market, mkt_mom.reindex(lvl.index))
        p_bear, p_neu, p_bull = _stub_probs(mom_eff, band)
        prob_df = pd.DataFrame({"bear": p_bear, "neutral": p_neu, "bull": p_bull})
        valid = mom_eff.notna()
        reg = pd.Series(index=prob_df.index, dtype=object)
        reg[valid] = prob_df[valid].idxmax(axis=1)

        for wend in lvl.index:
            if pd.isna(mom_eff.get(wend)):
                continue
            rows.append({
                "week_end_date": wend,
                "available_from": wend + BDay(publish_lag),  # §1-4 翌営業日から可用
                "anchor_id": str(sector),
                "p_bear": float(p_bear[wend]),
                "p_neutral": float(p_neu[wend]),
                "p_bull": float(p_bull[wend]),
                "regime": str(reg[wend]),
                "anchor_source": "market" if bool(use_market.get(wend, False)) else "sector",
                "n_constituents": int(cnt.get(wend, 0) or 0),
            })

    df = pd.DataFrame(rows, columns=[
        "week_end_date", "available_from", "anchor_id",
        "p_bear", "p_neutral", "p_bull", "regime",
        "anchor_source", "n_constituents",
    ])
    return validate_anchor(df)


# ============================================================ 本物の方向性HMM（step 5）
_ANCHOR_COLS = ["week_end_date", "available_from", "anchor_id",
                "p_bear", "p_neutral", "p_bull", "regime",
                "anchor_source", "n_constituents"]


def _wf_directional(r: np.ndarray, *, n_states: int, refit_every: int, window: int,
                    warmup: int, dof: float, sticky_kappa: float, n_init: int,
                    random_state: int) -> np.ndarray:
    """週足リターン(1D)に t-HMM を walk-forward、**平均リターンで整列**（state0=bear…last=bull）。

    終端 filtered（forward 正規化・≤t）＝オンライン・因果。返り値 (T, K)。
    """
    from .detector import _standardize_past, fit_best
    n = len(r)
    X = r.reshape(-1, 1)
    probs = np.full((n, n_states), np.nan)
    model = None
    for t in range(warmup, n):
        lo = max(0, t + 1 - window)
        win = _standardize_past(X[lo:t + 1])
        if len(win) < warmup or not np.isfinite(win).all():
            continue
        if model is None or (t - warmup) % refit_every == 0:
            model = fit_best(win, n_states=n_states, dof=dof, sticky_kappa=sticky_kappa,
                             n_init=n_init, random_state=random_state)
            model.align_by(model.means_[:, 0])           # 平均リターン昇順＝bear→bull
        probs[t] = model.filtered_proba(win)[-1]
    return probs


def build_weekly_anchor_hmm(
    adj_close: pd.DataFrame,
    s33: pd.Series,
    *,
    n_states: int = 3,
    freq: str = WEEKLY_FREQ,
    refit_every: int = 4,
    window: int = 156,
    warmup: int = 104,
    dof: float = 4.0,
    sticky_kappa: float = 5.0,
    n_init: int = 3,
    random_state: int = 42,
    min_constituents: int = FALLBACK_MIN_CONSTITUENTS,
    publish_lag: int = PUBLISH_LAG_BDAYS,
) -> pd.DataFrame:
    """本物の週足方向性HMM：セクター週足リターンに t-HMM walk-forward → bear/neutral/bull filtered。

    スタブ（build_weekly_anchor）と**同形**（同スキーマ・同 available_from ラグ・フォールバックB）。
    違いは regime が momentum スタブでなく **HMM filtered**（モメンタム整列・因果）である点。
    """
    levels, counts, mkt = _sector_indices(adj_close, s33)
    wk = levels.resample(freq).last()
    wk_ret = np.log(wk / wk.shift(1))
    wk_cnt = counts.resample(freq).last()
    mkt_wk = mkt.resample(freq).last()
    mkt_ret = np.log(mkt_wk / mkt_wk.shift(1))

    rows = []
    for sector in wk.columns:
        cnt_full = wk_cnt[sector]
        use_market_full = cnt_full < min_constituents
        r_eff = wk_ret[sector].where(~use_market_full, mkt_ret).dropna()
        if len(r_eff) < warmup + 10:
            continue
        idx_w = r_eff.index
        probs = _wf_directional(r_eff.to_numpy(), n_states=n_states, refit_every=refit_every,
                                window=window, warmup=warmup, dof=dof, sticky_kappa=sticky_kappa,
                                n_init=n_init, random_state=random_state)
        for i, wend in enumerate(idx_w):
            p = probs[i]
            if not np.isfinite(p).all():
                continue
            if n_states >= 3:
                p_bear, p_neu, p_bull = float(p[0]), float(p[1:-1].sum()), float(p[-1])
            else:
                p_bear, p_bull = float(p[0]), float(p[-1])
                p_neu = max(0.0, 1.0 - p_bear - p_bull)
            z = p_bear + p_neu + p_bull
            p_bear, p_neu, p_bull = p_bear / z, p_neu / z, p_bull / z
            reg = ("bear", "neutral", "bull")[int(np.argmax([p_bear, p_neu, p_bull]))]
            rows.append({
                "week_end_date": wend,
                "available_from": wend + BDay(publish_lag),
                "anchor_id": str(sector),
                "p_bear": p_bear, "p_neutral": p_neu, "p_bull": p_bull,
                "regime": reg,
                "anchor_source": "market" if bool(use_market_full.get(wend, False)) else "sector",
                "n_constituents": int(cnt_full.get(wend, 0) or 0),
            })
    return validate_anchor(pd.DataFrame(rows, columns=_ANCHOR_COLS))
