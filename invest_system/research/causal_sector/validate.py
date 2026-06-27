"""Purged K-Fold / CPCV と DSR による因果レジーム検証。

メタゲート（因果特徴量あり/なし）の OOS 比較と refutation 結果をまとめる。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from invest_system.validation.cpcv import CombinatorialPurgedKFold
from invest_system.validation.dsr import deflated_sharpe_ratio, sharpe_ratio


def causal_regime_cv_report(net_returns: pd.Series, features: pd.DataFrame, *,
                            n_splits: int = 6, n_test_splits: int = 2,
                            embargo_pct: float = 0.02,
                            fit_fn: Optional[Callable] = None) -> dict:
    """CPCV でメタ特徴量モデルの OOS Sharpe 分布を評価。

    Parameters
    ----------
    net_returns : 一次戦略の周期リターン
    features : 因果＋局面特徴量（PIT）
    fit_fn : (train_net, train_X, test_X) -> test_positions or sizes
             None なら単純に特徴量平均でサイズ化（ベースライン）。

    Returns
    -------
    dict with cpcv_sharpes, mean_sr, dsr, n_paths
    """
    net = net_returns.dropna()
    X = features.reindex(net.index).astype(float)
    common = net.index.intersection(X.dropna(how="all").index)
    net, X = net.loc[common], X.loc[common]
    if len(common) < n_splits * 2:
        return {"error": "insufficient samples", "n": len(common)}

    # t1 = ラベル終了（翌月）
    t1 = pd.Series(common[1:].tolist() + [common[-1]], index=common)

    cpcv = CombinatorialPurgedKFold(n_splits=n_splits,
                                    n_test_splits=n_test_splits,
                                    embargo_pct=embargo_pct)
    sharpes = []
    for train_idx, test_idx in cpcv.split(X, t1):
        tr_net = net.iloc[train_idx]
        te_net = net.iloc[test_idx]
        tr_X = X.iloc[train_idx]
        te_X = X.iloc[test_idx]
        if fit_fn is not None:
            sizes = fit_fn(tr_net, tr_X, te_X)
        else:
            # ベースライン：causal_edge が正ならフル、負なら半分
            ce = te_X.get("causal_edge", pd.Series(0.0, index=te_X.index))
            sizes = pd.Series(np.where(ce.fillna(0) > 0, 1.0, 0.5), index=te_X.index)
        gated = te_net * sizes.reindex(te_net.index).fillna(1.0)
        if gated.std() > 0 and len(gated) >= 3:
            sharpes.append(sharpe_ratio(gated))

    if not sharpes:
        return {"error": "no valid CPCV paths"}

    sr_arr = np.array(sharpes)
    dsr_val = None
    try:
        dsr_val = deflated_sharpe_ratio(
            sr_arr.mean(), sr_arr.std() ** 2,
            max(1, cpcv.get_n_paths()), len(net), 0.0, 3.0,
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "cpcv_sharpes": sharpes,
        "mean_sr": float(sr_arr.mean()),
        "std_sr": float(sr_arr.std()),
        "dsr": dsr_val,
        "n_paths": cpcv.get_n_paths(),
        "n_splits_config": f"{n_splits}/{n_test_splits}",
    }


def compare_gated_vs_ungated(primary_net: pd.Series,
                             base_features: pd.DataFrame,
                             causal_features: pd.DataFrame, *,
                             warmup: int = 36) -> pd.DataFrame:
    """因果特徴量追加前後の SR・DSR 比較（簡易）。"""
    from invest_system.research.meta_gate import fit_meta_gate

    idx = primary_net.dropna().index
    base = base_features.reindex(idx)
    full = base.join(causal_features.reindex(idx), how="left")

    def _sr_for(feat):
        size = fit_meta_gate(primary_net, feat, warmup=warmup)
        gated = primary_net * size.fillna(1.0)
        ok = gated.dropna()
        if ok.std() == 0 or len(ok) < 6:
            return np.nan, np.nan
        sr = sharpe_ratio(ok)
        try:
            dsr = deflated_sharpe_ratio(sr, 0.0, 2, len(ok), 0.0, 3.0)
        except Exception:  # noqa: BLE001
            dsr = np.nan
        return sr, dsr

    sr_b, dsr_b = _sr_for(base)
    sr_c, dsr_c = _sr_for(full)
    return pd.DataFrame([
        {"model": "base_meta", "sr": sr_b, "dsr": dsr_b},
        {"model": "causal_meta", "sr": sr_c, "dsr": dsr_c},
    ])