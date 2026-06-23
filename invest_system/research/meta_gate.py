"""メタラベル局面ゲート（二次モデル）：一次戦略の勝率を walk-forward 学習。

統合ナレッジベース §4.2 / DP4。AFML (López de Prado 2018) ch.3.6 / ch.10。
事前登録：docs/25-meta-gate-preregistration.md（FROZEN）。

一次モデル（value+PEAD 合成）が出すネット月次リターンを {0,1} メタラベル
（y_s = 1{net[s] > 0}）に変換し、各リバランス月 t で **t 以前のデータのみ**から
「翌月勝つ確率」を分類学習する。確率は AFML snippet 10.1（bet_size_from_prob）で
連続ベットサイズ ∈ [0,1] に変換し、`MetaGatedStrategy` が一次ウェイトに乗じる。

既存資産との差分：`RegimeGated`（離散ラベル×手設定 sizing）や
`walk_forward_regime_assignment`（単一レジーム条件付き平均でスリーブ離散選択）と異なり、
**複数 PIT 特徴量の確率分類→連続サイズ**を学習する＝AFML メタラベルの本体。

リーク防止（多重防御）：
  - 各 t の訓練は位置 ``≤ i-1-embargo`` の決済済みベットのみ（拡張窓・walk-forward）。
  - ``warmup`` 未満の訓練数ではゲート無効（NaN→ラッパで 1.0 扱い＝フル建玉）。
  - 当月の特徴量が欠損なら当月はゲートしない（NaN）。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..labeling.meta_labeling import bet_size_from_prob


def _default_model_factory():
    """標準化＋正則化ロジスティック回帰（FROZEN：docs/25 §3.3）。

    sklearn は遅延 import（本モジュールの import 自体は sklearn を要求しない）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=1000))


def meta_win_probability(primary_net: pd.Series, features: pd.DataFrame, *,
                         warmup: int = 36, embargo: int = 1,
                         model_factory: Optional[Callable[[], object]] = None
                         ) -> pd.Series:
    """各 t の P(来月 primary が勝つ | features[t]) を walk-forward 学習で返す。

    Parameters
    ----------
    primary_net : pd.Series
        一次戦略のネット周期リターン（index=リバランス日, net[s] は s→s+1 実現＝s+1 で既知）。
    features : pd.DataFrame
        局面特徴量（index=リバランス日, 各行は t 時点で PIT）。primary_net の index に整列される。
    warmup : int
        ゲート有効化に要する最小訓練サンプル数（既定 36＝3 年）。未満は NaN（ゲート無効）。
    embargo : int
        直近の決済済みベットを訓練から除外する本数（系列相関の遮断・既定 1）。
    model_factory : () -> estimator
        ``fit`` / ``predict_proba`` を持つ二次分類器の生成器。None で標準ロジスティック。

    Returns
    -------
    pd.Series
        各 t の予測勝率 p∈(0,1)。warmup 未達・特徴欠損は NaN。
    """
    if model_factory is None:
        model_factory = _default_model_factory
    net = primary_net.astype(float)
    idx = net.index
    X = features.reindex(idx).astype(float)
    y_all = (net > 0).astype(float)
    prob = pd.Series(np.nan, index=idx, dtype="float64")

    for i in range(len(idx)):
        train_hi = i - 1 - embargo            # この位置以下のラベルのみ使用（決済済み＋embargo）
        if train_hi < 0:
            continue
        xi = X.iloc[i]
        if xi.isna().any():
            continue                          # 当月の特徴欠損＝ゲートしない
        Xtr = X.iloc[:train_hi + 1]
        ytr = y_all.iloc[:train_hi + 1]
        ntr = net.iloc[:train_hi + 1]
        mask = Xtr.notna().all(axis=1) & ytr.notna() & ntr.notna()
        Xtr, ytr = Xtr[mask], ytr[mask]
        if len(ytr) < warmup:
            continue
        if ytr.nunique() < 2:                 # 単一クラス窓＝基準率を確率に
            prob.iloc[i] = float(np.clip(ytr.mean(), 1e-6, 1.0 - 1e-6))
            continue
        model = model_factory()
        model.fit(Xtr.to_numpy(), ytr.to_numpy())
        p = model.predict_proba(xi.to_numpy().reshape(1, -1))[0, 1]
        prob.iloc[i] = float(p)
    return prob


def fit_meta_gate(primary_net: pd.Series, features: pd.DataFrame, *,
                  warmup: int = 36, embargo: int = 1,
                  model_factory: Optional[Callable[[], object]] = None,
                  return_prob: bool = False) -> pd.Series:
    """walk-forward 勝率 → 連続ベットサイズ ∈ [0,1] を返す（AFML snippet 10.1）。

    `meta_win_probability` の確率を `bet_size_from_prob` でサイズ化し [0,1] にクリップ
    （勝率<0.5 の月は 0 方向＝建玉を絞る）。NaN（warmup 未達・特徴欠損）はそのまま返し、
    `MetaGatedStrategy` が 1.0（ゲート無効＝フル建玉）として扱う。

    return_prob=True なら（size, prob）の DataFrame（診断用）。
    """
    prob = meta_win_probability(primary_net, features, warmup=warmup,
                                embargo=embargo, model_factory=model_factory)
    size = pd.Series(np.nan, index=prob.index, dtype="float64")
    ok = prob.notna()
    if ok.any():
        size[ok] = np.clip(bet_size_from_prob(prob[ok].to_numpy()), 0.0, 1.0)
    if return_prob:
        return pd.concat({"size": size, "prob": prob}, axis=1)
    return size
