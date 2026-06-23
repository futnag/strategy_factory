"""メタラベル局面ゲート（二次モデル）の検証。ネットワーク不要・合成データ。

事前登録：docs/25-meta-gate-preregistration.md。
検証点：(1) walk-forward の先読み不能、(2) 学習が局面情報を捉える、(3) warmup 挙動、
(4) MetaGatedStrategy の恒等/退化/スケール適用、(5) スケールの PIT 適用。
"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.data_view import AsOfView
from invest_system.research.meta_gate import fit_meta_gate, meta_win_probability
from invest_system.research.strategy import MetaGatedStrategy, Strategy


def _series(values, start="2016-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


class _FixedStrategy(Strategy):
    """常に固定ウェイトを返す一次戦略のスタブ（ゲート適用の検証用）。"""

    def __init__(self, weights: pd.Series, name="fixed"):
        self.w = weights
        self.name = name
        self.params = {"kind": "fixed"}

    def target_weights(self, asof):
        return self.w.copy()


# --- fit_meta_gate / meta_win_probability：walk-forward の先読み不能 -----------
def test_gate_is_walk_forward_no_lookahead():
    rng = np.random.default_rng(0)
    n = 120
    feat = _series(rng.normal(size=n))
    # ラベルは feat の符号に弱く連動（学習可能な構造）＋ノイズ
    net = _series(np.where(feat.to_numpy() > 0, 1.0, -1.0) * 0.02
                  + rng.normal(scale=0.01, size=n))
    features = pd.DataFrame({"f": feat})

    base = fit_meta_gate(net, features, warmup=24, embargo=1)

    # 位置 p 以降の net と features を改変しても、scale[:p] は不変であるべき
    p = 80
    net2 = net.copy()
    net2.iloc[p:] = -net2.iloc[p:]          # 未来のラベルを反転
    features2 = features.copy()
    features2.iloc[p:, 0] = -features2.iloc[p:, 0]
    after = fit_meta_gate(net2, features2, warmup=24, embargo=1)

    pd.testing.assert_series_equal(base.iloc[:p], after.iloc[:p])


def test_gate_learns_regime_separation():
    # feat>0 の月は一次が勝ち、feat<=0 は負け、という明快な構造を学習できるか
    rng = np.random.default_rng(1)
    n = 160
    feat = _series(rng.normal(size=n))
    net = _series(np.where(feat.to_numpy() > 0, 1.0, -1.0) * 0.03
                  + rng.normal(scale=0.005, size=n))
    features = pd.DataFrame({"f": feat})

    out = fit_meta_gate(net, features, warmup=36, embargo=1, return_prob=True)
    active = out.dropna()
    assert len(active) > 0
    # 予測勝率：feat>0 の月は feat<=0 の月より明確に高い
    hi = active.loc[feat.reindex(active.index) > 0, "prob"].mean()
    lo = active.loc[feat.reindex(active.index) <= 0, "prob"].mean()
    assert hi > lo + 0.15
    # サイズは [0,1]、有利月の平均サイズ > 不利月
    assert ((active["size"] >= 0.0) & (active["size"] <= 1.0)).all()
    s_hi = active.loc[feat.reindex(active.index) > 0, "size"].mean()
    s_lo = active.loc[feat.reindex(active.index) <= 0, "size"].mean()
    assert s_hi > s_lo


def test_gate_warmup_and_missing_are_nan():
    n = 60
    feat = _series(np.linspace(-1, 1, n))
    net = _series(np.r_[np.ones(30), -np.ones(30)] * 0.02)
    features = pd.DataFrame({"f": feat})
    size = fit_meta_gate(net, features, warmup=40, embargo=1)
    # warmup 未達の先頭はゲート無効（NaN）
    assert size.iloc[:40].isna().all()
    # 特徴欠損の月もゲートしない（NaN）
    features.iloc[50, 0] = np.nan
    size2 = fit_meta_gate(net, features, warmup=40, embargo=1)
    assert np.isnan(size2.iloc[50])


def test_single_class_window_uses_base_rate():
    # 全勝の窓 → 確率は基準率（≈1）→ サイズ正（建玉を維持）
    n = 80
    feat = _series(np.zeros(n))              # 無情報特徴
    net = _series(np.ones(n) * 0.01)         # 常に勝ち（単一クラス）
    features = pd.DataFrame({"f": feat})
    out = fit_meta_gate(net, features, warmup=24, embargo=1, return_prob=True)
    active = out.dropna()
    assert (active["prob"] > 0.9).all()
    assert (active["size"] > 0.5).all()


# --- MetaGatedStrategy：適用の検証 ------------------------------------------
def _view_and_weights():
    idx = pd.date_range("2024-01-31", periods=4, freq="ME")
    close = pd.DataFrame({"A": [10.] * 4, "B": [20.] * 4}, index=idx)
    return AsOfView({"close": close}), idx, pd.Series({"A": 0.5, "B": -0.5})


def test_metagate_identity_when_scale_one():
    view, idx, w = _view_and_weights()
    scale = pd.Series(1.0, index=idx)
    gated = MetaGatedStrategy(_FixedStrategy(w), scale)
    out = gated.target_weights(view.asof(idx[-1]))
    pd.testing.assert_series_equal(out.sort_index(), w.sort_index())


def test_metagate_half_scales_weights():
    view, idx, w = _view_and_weights()
    gated = MetaGatedStrategy(_FixedStrategy(w), pd.Series(0.5, index=idx))
    out = gated.target_weights(view.asof(idx[-1]))
    pd.testing.assert_series_equal(out.sort_index(), (w * 0.5).sort_index())


def test_metagate_zero_scale_is_cash():
    view, idx, w = _view_and_weights()
    gated = MetaGatedStrategy(_FixedStrategy(w), pd.Series(0.0, index=idx))
    assert gated.target_weights(view.asof(idx[-1])).empty


def test_metagate_nan_scale_defaults_to_full():
    view, idx, w = _view_and_weights()
    scale = pd.Series([np.nan, np.nan, np.nan, np.nan], index=idx)
    gated = MetaGatedStrategy(_FixedStrategy(w), scale)
    out = gated.target_weights(view.asof(idx[-1]))       # 未確定＝フル建玉
    pd.testing.assert_series_equal(out.sort_index(), w.sort_index())


def test_metagate_uses_latest_scale_pit():
    view, idx, w = _view_and_weights()
    # スケールは時刻ごとに異なる。t での適用は「≤t の最新」のみ
    scale = pd.Series([0.2, 0.4, 0.6, 0.8], index=idx)
    gated = MetaGatedStrategy(_FixedStrategy(w), scale)
    out = gated.target_weights(view.asof(idx[1]))        # idx[1] では 0.4
    pd.testing.assert_series_equal(out.sort_index(), (w * 0.4).sort_index())
