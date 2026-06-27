"""DoWhy + EconML による因果効果推定と refutation test。

因果グラフから同定した adjustment set で backdoor adjustment を行い、
VALUE→RET の ATE/CATE を推定する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .graph import CausalGraphResult


@dataclass
class CausalEffectResult:
    """因果効果推定結果。"""
    ate: float
    ate_se: Optional[float]
    cate_mean: Optional[float]
    adjustment_vars: list[str]
    colliders_excluded: list[str]
    refutation: dict[str, float] = field(default_factory=dict)
    method: str = "backdoor"


def _build_adjustment_df(panel: pd.DataFrame, adj_vars: list[str],
                         treatment: str = "VALUE",
                         outcome: str = "RET") -> pd.DataFrame:
    """調整変数をラグ1でシフト（PIT）。"""
    df = pd.DataFrame(index=panel.index)
    df["treatment"] = panel[treatment]
    df["outcome"] = panel[outcome].shift(-1)  # 翌日リターン（因果的ラベル）
    for v in adj_vars:
        if v in panel.columns:
            df[f"adj_{v}"] = panel[v].shift(1)
    return df.dropna()


def estimate_causal_effect(panel: pd.DataFrame, graph: CausalGraphResult, *,
                           treatment: str = "VALUE", outcome: str = "RET",
                           use_econml: bool = True) -> CausalEffectResult:
    """DoWhy backdoor + オプションで EconML CATE。"""
    adj_vars = graph.adjustment_set.get("VALUE_to_RET", [])
    colliders = graph.colliders
    # collider を明示除外
    adj_vars = [v for v in adj_vars if v not in colliders
                and v not in (treatment, outcome)]
    df = _build_adjustment_df(panel, adj_vars, treatment, outcome)
    if len(df) < 100:
        raise ValueError(f"insufficient rows for causal effect: {len(df)}")

    adj_cols = [c for c in df.columns if c.startswith("adj_")]
    common_causes = ", ".join(adj_cols) if adj_cols else ""

    from dowhy import CausalModel

    # common_causes のみで backdoor 同定（グラフ文字列はノード名不一致を避ける）
    model = CausalModel(
        data=df,
        treatment="treatment",
        outcome="outcome",
        common_causes=adj_cols if adj_cols else None,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )
    ate = float(estimate.value)
    ate_se = None
    try:
        ate_se = float(estimate.get_standard_error())
    except Exception:  # noqa: BLE001
        pass

    cate_mean = None
    if use_econml and adj_cols:
        try:
            from econml.dml import LinearDML
            from sklearn.linear_model import LassoCV, RidgeCV

            T = df["treatment"].values.reshape(-1, 1)
            Y = df["outcome"].values
            X = df[adj_cols].values if adj_cols else np.zeros((len(df), 1))
            W = np.zeros((len(df), 1))
            est = LinearDML(model_y=RidgeCV(), model_t=LassoCV(),
                            discrete_treatment=False, cv=3)
            est.fit(Y, T, X=X, W=W)
            cate = est.effect(X)
            cate_mean = float(np.mean(cate))
        except Exception:  # noqa: BLE001
            pass

    return CausalEffectResult(
        ate=ate, ate_se=ate_se, cate_mean=cate_mean,
        adjustment_vars=adj_vars, colliders_excluded=colliders,
        method="backdoor.linear_regression",
    )


def run_refutation_tests(panel: pd.DataFrame, graph: CausalGraphResult, *,
                         treatment: str = "VALUE", outcome: str = "RET",
                         n_simulations: int = 50) -> dict[str, float]:
    """DoWhy refutation: placebo treatment, random common cause。"""
    adj_vars = [v for v in graph.adjustment_set.get("VALUE_to_RET", [])
                if v not in graph.colliders]
    df = _build_adjustment_df(panel, adj_vars, treatment, outcome)
    adj_cols = [c for c in df.columns if c.startswith("adj_")]
    if len(df) < 100:
        return {}

    import dowhy
    from dowhy import CausalModel

    model = CausalModel(
        data=df, treatment="treatment", outcome="outcome",
        common_causes=adj_cols if adj_cols else None,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified,
                                     method_name="backdoor.linear_regression")

    refutations = {}
    try:
        r_placebo = model.refute_estimate(
            identified, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute", num_simulations=n_simulations,
        )
        refutations["placebo_pvalue"] = float(r_placebo.refutation_result["p_value"])
    except Exception:  # noqa: BLE001
        refutations["placebo_pvalue"] = np.nan

    try:
        r_random = model.refute_estimate(
            identified, estimate,
            method_name="random_common_cause",
            num_simulations=n_simulations,
        )
        refutations["random_cause_pvalue"] = float(
            r_random.refutation_result.get("p_value", np.nan))
    except Exception:  # noqa: BLE001
        refutations["random_cause_pvalue"] = np.nan

    try:
        r_sub = model.refute_estimate(
            identified, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8, num_simulations=10,
        )
        refutations["subset_stability"] = float(
            r_sub.refutation_result.get("estimated_effect", np.nan))
    except Exception:  # noqa: BLE001
        refutations["subset_stability"] = np.nan

    return refutations