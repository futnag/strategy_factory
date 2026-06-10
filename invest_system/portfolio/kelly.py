"""ケリー基準（連続版）— 実運用レバレッジの導出（L9/Phase 2・Chan 補完・DP16）。

Chan『Quantitative Trading』ch.6 /『Algorithmic Trading』ch.8。ガウス近似の連続版
ケリーでは、単一戦略の最適レバレッジは f* = μ/σ²（周期次元は約分されるので年率/周期
どちらで計算しても同値）、複数戦略では F* = C⁻¹M（C=共分散, M=平均超過リターン）。

満額ケリーは禁止（DP16）。理由：①μ/σ² は推定誤差に二乗で敏感（とくに μ）②日本株は
値幅制限・決算/日銀イベントのギャップで「連続リバランス可能」という連続版の前提が
崩れる ③ファットテール下で破滅リスクが残る。既定 fraction=0.5（ハーフケリー）、
保守は 0.25。複数戦略版の C はノイズ除去（denoise.denoise_covariance）してから渡す。

ここで出すのは**研究値**（バックテスト系列からの推定）。実運用では実測リターンで
月次再推定し、レバレッジ上限（ハード限度）と併用する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KellyResult:
    f_full: float        # 満額ケリー f* = μ/σ²（最適レバレッジ・推定値）
    f: float             # fraction 適用後の推奨レバレッジ（= fraction × f*）
    fraction: float      # 使用したケリー分数（0.5=ハーフ, 0.25=クォーター）
    mu_ann: float        # 年率平均リターン（算術）
    sigma_ann: float     # 年率ボラティリティ
    growth_ann_full: float   # 満額時の期待対数成長率（年率, g(f*)=μ²/2σ²）
    growth_ann_frac: float   # fraction 適用時の期待対数成長率（年率）
    ann_factor: float    # 年率換算に使った周期数/年

    def summary(self) -> str:
        return (f"f*={self.f_full:.2f} → f({self.fraction:g}×)={self.f:.2f}  "
                f"[μ={self.mu_ann:+.1%}/σ={self.sigma_ann:.1%} 年率, "
                f"g={self.growth_ann_frac:+.1%}/満額{self.growth_ann_full:+.1%}]")


def _infer_ann_factor(idx) -> float:
    """DatetimeIndex から周期数/年を推定（engine._ann_factor と同方式）。"""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return float("nan")
    years = (idx[-1] - idx[0]).days / 365.25
    return float(len(idx) / years) if years > 0 else float("nan")


def kelly_fraction(returns: pd.Series, *, fraction: float = 0.5,
                   ann_factor: float | None = None) -> KellyResult:
    """周期リターン系列から（フラクショナル）ケリー・レバレッジを推定する。

    returns: 超過リターンの周期系列（バックテストのネット系列等。無リスク金利を
      引いた系列を渡すのが厳密だが、現行の低金利では近似的に省略可）。
    ann_factor: 周期数/年。None なら DatetimeIndex から推定（月次≈12, 日次≈252）。
    f* が負（平均が負）の場合もそのまま返す＝「賭けるな」という推定として解釈する。
    """
    r = returns.dropna()
    if len(r) < 8:
        raise ValueError("kelly_fraction: 観測が少なすぎる（8未満）")
    if ann_factor is None:
        ann_factor = _infer_ann_factor(r.index)
        if not np.isfinite(ann_factor):
            raise ValueError("kelly_fraction: ann_factor を推定できない（明示指定を）")
    mu_p = float(r.mean())
    var_p = float(r.var(ddof=1))
    if var_p <= 0:
        raise ValueError("kelly_fraction: 分散が 0")
    f_full = mu_p / var_p                       # 周期次元は約分される
    f = fraction * f_full
    growth = lambda x: x * mu_p - 0.5 * x * x * var_p   # noqa: E731  g(f)/周期
    return KellyResult(
        f_full=float(f_full), f=float(f), fraction=float(fraction),
        mu_ann=mu_p * ann_factor, sigma_ann=float(np.sqrt(var_p * ann_factor)),
        growth_ann_full=growth(f_full) * ann_factor,
        growth_ann_frac=growth(f) * ann_factor, ann_factor=float(ann_factor))


def kelly_weights(mu, cov, *, fraction: float = 0.5):
    """複数戦略の連続版ケリー配分 F = fraction × C⁻¹M（Chan AT ch.8）。

    mu: 各戦略の周期平均超過リターン（Series/array）。cov: 同周期の共分散
    （DataFrame/array, **denoise_covariance でノイズ除去してから**渡すこと）。
    返り値は各戦略へのレバレッジ（合計は 1 に正規化しない＝グロスがレバレッジ）。
    """
    is_pd = isinstance(mu, pd.Series)
    m = mu.to_numpy(dtype=float) if is_pd else np.asarray(mu, dtype=float)
    c = cov.to_numpy(dtype=float) if isinstance(cov, pd.DataFrame) \
        else np.asarray(cov, dtype=float)
    f = fraction * np.linalg.solve(c, m)
    return pd.Series(f, index=mu.index) if is_pd else f
