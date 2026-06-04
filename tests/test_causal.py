"""因果フィルタ：コライダーバイアスの符号反転と LiNGAM 方向判定を検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.features.causal import (
    causal_filter,
    classify_features,
    collider_bias_beta,
    direction_score,
)

_S = 1.0 / np.sqrt(2.0)   # 単位分散ラプラス（var = 2*scale²）の scale


def test_collider_bias_formula_value():
    # (β − δγ)/(1+γ²) = (0.5 − 2)/2 = −0.75（符号反転）
    assert collider_bias_beta(0.5, 2.0, 1.0) == pytest.approx(-0.75)


def test_collider_bias_matches_regression():
    rng = np.random.default_rng(0)
    n = 80000
    x = rng.laplace(0, _S, n)
    eps = rng.laplace(0, _S, n)
    zeta = rng.laplace(0, _S, n)
    beta, gamma, delta = 0.5, 1.0, 2.0
    y = beta * x + eps
    z = gamma * y + delta * x + zeta

    b_simple = np.cov(x, y)[0, 1] / np.var(x)            # y~x
    W = np.column_stack([x, z, np.ones(n)])
    b_x = np.linalg.lstsq(W, y, rcond=None)[0][0]        # y~[x,z]

    assert b_simple == pytest.approx(beta, abs=0.03)     # 単独回帰は +0.5
    assert b_x == pytest.approx(collider_bias_beta(beta, delta, gamma), abs=0.03)
    assert b_x < 0                                        # 合流点制御で符号反転


def test_direction_score_cause_positive():
    rng = np.random.default_rng(0)
    x = rng.laplace(0, _S, 5000)
    y = 0.8 * x + 0.4 * rng.laplace(0, _S, 5000)          # x → y
    assert direction_score(x, y) > 0


def test_direction_score_effect_negative():
    rng = np.random.default_rng(1)
    y = rng.laplace(0, _S, 5000)
    x = 0.8 * y + 0.4 * rng.laplace(0, _S, 5000)          # y → x（x は結果）
    assert direction_score(x, y) < 0


def test_classify_and_filter():
    rng = np.random.default_rng(2)
    n = 6000
    cause = rng.laplace(0, _S, n)
    y = 0.8 * cause + 0.4 * rng.laplace(0, _S, n)
    effect = 0.7 * y + 0.4 * rng.laplace(0, _S, n)         # y の下流＝コライダー
    X = pd.DataFrame({"cause": cause, "effect": effect})

    cls = classify_features(X, pd.Series(y))
    assert cls.loc["cause", "role"] == "cause"
    assert cls.loc["effect", "role"] == "effect"

    filtered, _ = causal_filter(X, pd.Series(y))
    assert list(filtered.columns) == ["cause"]            # コライダーを除去
