"""ポートフォリオ層（L9）：共分散ノイズ除去・ロバスト配分・レバレッジ導出。

設計書 §6 / DP9, DP16。AFML ch.2（ノイズ除去）, ch.16（HRP）, NCO。
逆行列を直接使う Markowitz の「誤差最大化」を、RMT ノイズ除去・HRP・NCO で緩和する。
実運用レバレッジはフラクショナル・ケリー（kelly.py・Chan 補完・DP16）で上限を導出する。
"""

from .allocation import hrp_weights, min_variance_weights, nco_weights
from .denoise import cov_to_corr, denoise_covariance, marcenko_pastur_pdf
from .kelly import KellyResult, kelly_fraction, kelly_weights

__all__ = [
    "cov_to_corr",
    "denoise_covariance",
    "hrp_weights",
    "KellyResult",
    "kelly_fraction",
    "kelly_weights",
    "marcenko_pastur_pdf",
    "min_variance_weights",
    "nco_weights",
]
