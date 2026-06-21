"""Phase 4 推定器スイート：線形ベースライン → 木 → 浅い NN（拡張窓・内側CV・CPCV-OOS）。

handoff §5：拡張窓 train/validate/test。ハイパラは**内側 CV（train 内）でのみ**決定し、
テスト/OOS フォールドではチューニングしない。線形は明示的なマクロ交互作用列を使い、木/NN は
生の特性＋マクロ（交互作用はモデルが学習）。seed 固定・決定的（再現性）。既存 validation を再利用。

このモジュールは「予測の生成」だけを担い、経済評価（IC/L-S/手数料後/封筒）と判定は
phase4_evaluate.py / run_phase4_judgment.py が行う（throwaway 診断と一度きり判定の分離）。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..equities.design_matrix import DesignMatrix, add_interactions

SEED = 20260621
LINEAR = ("ols", "lasso", "elasticnet")
TREE = ("rf", "hgbrt")
NN = ("mlp",)
ALL_FAMILIES = LINEAR + TREE + NN


# --- モデル工場（決定的・内側CVは fit 内で train のみ使用） -----------------
def make_model(family: str):
    """family→ sklearn 推定器（線形/NN は StandardScaler 同梱の Pipeline）。

    内側 CV：Lasso/ElasticNet は *CV 版（fit 時に train 内で alpha/l1_ratio を選択）。
    木/NN は early stopping（HGBRT/MLP の内部 validation_fraction）で過学習を抑制＝
    テストを覗かずに複雑度を決める。ハイパラ範囲は事前固定（探索しない）。
    """
    from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if family == "ols":
        return Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    if family == "lasso":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LassoCV(cv=5, random_state=SEED, max_iter=5000))])
    if family == "elasticnet":
        return Pipeline([("sc", StandardScaler()),
                         ("m", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5,
                                            random_state=SEED, max_iter=5000))])
    if family == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=8,
                                     min_samples_leaf=200, max_features="sqrt",
                                     n_jobs=-1, random_state=SEED)
    if family == "hgbrt":
        return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                             max_iter=300, l2_regularization=1.0,
                                             early_stopping=True, validation_fraction=0.15,
                                             random_state=SEED)
    if family == "mlp":
        return Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(32, 8), alpha=1e-3,
                                            max_iter=300, early_stopping=True,
                                            validation_fraction=0.15, random_state=SEED))])
    raise ValueError(f"unknown family: {family}")


def needs_interactions(family: str) -> bool:
    """線形のみ明示交互作用列（木/NN は生入力でモデルが交互作用を学習＝次元爆発回避）。"""
    return family in LINEAR


# --- 設計行列 → モデル入力（family 別の列構成） ----------------------------
def make_inputs(D: DesignMatrix, family: str) -> tuple[pd.DataFrame, pd.Series]:
    """family に応じた X 行列と y を返す。マクロは行の月で broadcast、欠損は ffill/bfill/0。

    線形：[chars | macro | char×macro]。木/NN：[chars | macro]。chars は rank_and_fill 済で
    欠損無し。macro（状態）は時系列 ffill→bfill→0 で穴埋め（先頭被覆外のみ＝OOS 窓外）。
    """
    macro = D.macro.ffill().bfill().fillna(0.0)
    dates = D.X.index.get_level_values("date")
    macro_long = pd.DataFrame(macro.reindex(dates).to_numpy(), index=D.X.index,
                              columns=D.macro_cols)
    chars = D.X[D.char_cols]
    if needs_interactions(family):
        X = add_interactions(chars, macro_long, D.char_cols, D.macro_cols)
    else:
        X = pd.concat([chars, macro_long], axis=1)
    return X, D.y


# --- 月グループ CPCV（時間軸で purge/embargo・特徴量窓考慮） ----------------
def _month_groups(n_months: int, n_splits: int) -> list[list[int]]:
    return [[int(i) for i in g] for g in np.array_split(np.arange(n_months), n_splits)]


def purge_train_positions(n_months: int, test_pos, gap_before: int,
                          gap_after: int) -> list[int]:
    """テスト位置集合から train 月位置を返す。情報窓 [m−gap_after, m+gap_before] が

    テストに重なる train obs を落とす：`gap_before`＝ラベル前向き重複（1M）、`gap_after`＝
    特徴量の後ろ向き窓（mom_36m 等）＋系列相関の後方遮断（embargo）。テスト自身も除外。
    """
    test_set = set(int(t) for t in test_pos)
    drop = set(test_set)
    for t in test_set:
        for m in range(t - gap_before, t + gap_after + 1):
            if 0 <= m < n_months:
                drop.add(m)
    return [i for i in range(n_months) if i not in drop]


def cpcv_paths_predict(D: DesignMatrix, family: str, n_splits: int = 6,
                       n_test_splits: int = 2, gap_before: int = 1,
                       gap_after: int = 12, model_factory: Optional[Callable] = None
                       ) -> list[pd.Series]:
    """Combinatorial Purged CV：φ=C(N-1,k-1) 本の完全 OOS 予測パスを返す（各 (date,Code) 1回/パス）。

    C(N,k) 分割を回し、各 train でテスト群を予測。各群はちょうど φ 分割でテストされるので、
    群 g の φ 個の予測を path 0..φ-1 に割り当て→各 path が全群を 1 度ずつ覆う完全 OOS 系列。
    purge/embargo は purge_train_positions（特徴量窓 gap_after・ラベル gap_before）。
    """
    X, y = make_inputs(D, family)
    months = D.months
    date = X.index.get_level_values("date")
    mf = model_factory or (lambda: make_model(family))
    groups = _month_groups(len(months), n_splits)
    group_preds: dict[int, list[pd.Series]] = {g: [] for g in range(n_splits)}
    for combo in itertools.combinations(range(n_splits), n_test_splits):
        test_pos = sorted(p for g in combo for p in groups[g])
        train_pos = purge_train_positions(len(months), test_pos, gap_before, gap_after)
        tr = X.index[date.isin(months[train_pos])]
        if len(tr) < 100:
            continue
        model = mf()
        model.fit(X.loc[tr].to_numpy(), y.loc[tr].to_numpy())
        for g in combo:
            te = X.index[date.isin(months[groups[g]])]
            if len(te):
                group_preds[g].append(pd.Series(model.predict(X.loc[te].to_numpy()),
                                                index=te))
    n_paths = min((len(v) for v in group_preds.values()), default=0)
    paths = []
    for j in range(n_paths):
        paths.append(pd.concat([group_preds[g][j] for g in range(n_splits)]).sort_index())
    return paths


def consensus(paths: list[pd.Series]) -> pd.Series:
    """φ 本のパス予測の (date,Code) 平均＝consensus OOS 予測（R²/IC 用）。"""
    if not paths:
        return pd.Series(dtype="float64")
    return pd.concat(paths, axis=1).mean(axis=1)


# --- 拡張窓 OOS 予測（厳密に因果・開発の R²/増分用） -----------------------
def expanding_oos_predict(D: DesignMatrix, family: str, min_train_months: int = 36,
                          refit_every: int = 12, embargo_months: int = 1,
                          model_factory: Optional[Callable] = None) -> pd.Series:
    """拡張窓：各リフィット時点で「過去のみ」で学習し以後 refit_every 月を予測（厳密に因果）。

    train は月 ≤ (テスト開始 − 1 − embargo)。テスト/OOS ではチューニングしない（内側 CV は
    fit 内 train のみ）。GKX 流 OOS R² と線形 vs ML 増分の算出に使う（CPCV と別経路）。
    """
    X, y = make_inputs(D, family)
    mf = model_factory or (lambda: make_model(family))
    months = D.months
    date = X.index.get_level_values("date")
    oos = pd.Series(np.nan, index=X.index)
    start = min_train_months
    while start < len(months):
        tr_end = start - embargo_months              # embargo：直前 embargo 月を学習から除く
        if tr_end < min_train_months // 2:
            start += refit_every
            continue
        tr_months = months[:tr_end]
        te_months = months[start:start + refit_every]
        tr = X.index[date.isin(tr_months)]
        te = X.index[date.isin(te_months)]
        if len(tr) >= 100 and len(te) > 0:
            model = mf()
            model.fit(X.loc[tr].to_numpy(), y.loc[tr].to_numpy())
            oos.loc[te] = model.predict(X.loc[te].to_numpy())
        start += refit_every
    return oos


@dataclass
class FamilyOOS:
    family: str
    oos: pd.Series              # OOS 予測（index=(date,Code)）
    mode: str                   # "cpcv" | "expanding"
