"""Phase 4 推定器プラミングの検証（合成・ネットワーク不要・決定的）。

month_cpcv の purge/embargo 境界、make_inputs の交互作用次元、make_model の構築を固定する。
実データの拡張窓/CPCV 予測は run_phase4_judgment の実行で確認（重い）。
"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.design_matrix import DesignMatrix
from invest_system.research import phase4_estimators as est


def _synthetic_D(n_months=12, codes=("A", "B", "C", "D")):
    months = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    idx = pd.MultiIndex.from_product([months, list(codes)], names=["date", "Code"])
    rng = np.random.default_rng(0)
    char_cols = ["f1", "f2", "f3"]
    X = pd.DataFrame(rng.uniform(-1, 1, (len(idx), 3)), index=idx, columns=char_cols)
    y = pd.Series(rng.normal(0, 0.05, len(idx)), index=idx)
    macro_cols = ["m1", "m2"]
    macro = pd.DataFrame(rng.normal(0, 1, (n_months, 2)), index=months, columns=macro_cols)
    t1 = pd.Series(months, index=months).shift(-1)
    t1.iloc[-1] = months[-1]
    uni = pd.DataFrame(True, index=months, columns=list(codes))
    return DesignMatrix(X=X, y=y, macro=macro, months=months, t1=t1,
                        char_cols=char_cols, macro_cols=macro_cols, universe=uni)


# --- 交互作用次元（線形のみ明示・木/NN は生入力） --------------------------
def test_make_inputs_linear_has_interactions():
    D = _synthetic_D()
    Xl, yl = est.make_inputs(D, "lasso")
    # 線形：chars(3) + macro(2) + 交互作用(3×2=6) = 11
    assert Xl.shape[1] == 3 + 2 + 6
    assert any("__x__" in c for c in Xl.columns)
    assert len(yl) == len(D.y)


def test_make_inputs_tree_no_interactions():
    D = _synthetic_D()
    Xt, _ = est.make_inputs(D, "rf")
    assert Xt.shape[1] == 3 + 2                      # chars + macro のみ（交互作用なし）
    assert not any("__x__" in c for c in Xt.columns)
    # マクロは行の月で broadcast されている
    assert (Xt.groupby(level="date")["m1"].nunique() == 1).all()


# --- purge/embargo 境界：ラベル前向き＋特徴量後ろ向き窓を落とす ------------
def test_purge_train_positions_label_and_feature_window():
    n = 12
    test_pos = [5, 6]                               # テストブロック＝月 5-6
    train = set(est.purge_train_positions(n, test_pos, gap_before=1, gap_after=3))
    assert 5 not in train and 6 not in train        # テスト自身は除外
    assert 4 not in train                           # gap_before=1：ラベル[4,5]がテストに重なる
    assert {7, 8, 9}.isdisjoint(train)              # gap_after=3：特徴量窓がテストに届く後続3月を除去
    assert 10 in train                              # gap_after を超えた月は学習に残る
    assert 0 in train and 3 in train                # テスト前（重複なし）は残る


def test_purge_train_positions_long_feature_window_36():
    # 最長特徴量窓 36ヶ月を gap_after=36 で確実に除去（mom_36m リーク遮断・docs/19 ★）
    n = 120
    train = set(est.purge_train_positions(n, [60], gap_before=1, gap_after=36))
    assert {61, 80, 96}.isdisjoint(train)           # テスト後 36ヶ月以内は全除去
    assert 97 in train                              # 36ヶ月超は残る


# --- CPCV：φ パス再構成（各パスが全 (date,Code) を一度ずつ覆う） -----------
def test_cpcv_paths_reconstruction_covers_all():
    from math import comb
    D = _synthetic_D(n_months=12, codes=tuple(f"C{i}" for i in range(40)))
    paths = est.cpcv_paths_predict(D, "ols", n_splits=6, n_test_splits=2,
                                   gap_before=1, gap_after=1)
    assert len(paths) == comb(6 - 1, 2 - 1)         # φ = C(5,1) = 5 パス
    full = set(D.X.index)
    for p in paths:
        assert set(p.index) == full                 # 各パスが全 (date,Code) を覆う
        assert not p.index.duplicated().any()       # 重複なし（各観測1回/パス）
        assert np.isfinite(p.to_numpy()).all()


# --- モデル工場が構築でき fit/predict 可能（決定的） ----------------------
@pytest.mark.parametrize("fam", est.ALL_FAMILIES)
def test_make_model_builds_and_fits(fam):
    m = est.make_model(fam)
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (200, 5))
    y = X[:, 0] * 0.3 + rng.normal(0, 0.1, 200)
    m.fit(X, y)
    p = m.predict(X[:10])
    assert len(p) == 10 and np.isfinite(p).all()
