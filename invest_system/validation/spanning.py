"""推定リスク込みの OOS Sharpe 分布とスパニング型増分判定（KWZ 2024）。

Kan, Wang & Zheng (2024) "In-sample and out-of-sample Sharpe ratios of multi-factor
asset pricing models", JFE 155, 103837 の実装（H-14 の出口＝改良研究の増分判定）。
仕様と数値検証は research_loop/surveys/2026-07-04-methodology-youngsample.md §A T1。

中核事実: 推定ウェイトの接線ポートフォリオの (θ̂_IS, θ̃_OOS) の同時分布は
(N, T, θ) のみに依存する（Prop.1）。凍結ウェイトのベースライン（推定リスクゼロ）に
候補スリーブを加えた N=2 の接線が「ベースラインを OOS で上回るか」を、
break-even θ_b（期待ショートフォール基準）とシフト帰無 GRS で判定できる。

用途（DP18 に従い**表示・提案専用**＝judge の合否には使わない。採用は人間ゲート）:
  - 改良研究: 「変種 B をベースライン A に加えた接線の母集団 SR が θ_b を超えるか」
  - minTRL の代替視点: 推定リスク込みの期待 OOS SR の見積り
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats


def simulate_sharpes(n_assets: int, t_obs: int, theta: float, *, n_sims: int = 200_000,
                     rng: np.random.Generator | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(θ̂_IS, θ̃_OOS) の同時サンプル（KWZ Prop.1・iid 正規仮定）。

    theta は対象期間頻度（例: 月次）の真の接線 Sharpe。返り値も同頻度。
    """
    if n_assets < 2:
        raise ValueError("N>=2（ベースライン+候補で最低2）")
    if t_obs <= n_assets + 2:
        raise ValueError("T > N+2 が必要")
    rng = rng or np.random.default_rng(0)
    n, t = n_assets, t_obs
    u1 = rng.chisquare(t - n, n_sims)
    b = rng.beta((t - n + 1) / 2.0, (n - 1) / 2.0, n_sims)
    z = rng.normal(np.sqrt(b * t) * theta, 1.0)
    u = rng.noncentral_chisquare(n - 1, np.maximum((1.0 - b) * t * theta ** 2, 1e-12),
                                 n_sims) if n > 1 else np.zeros(n_sims)
    theta_is = np.sqrt((z ** 2 + u) / u1)
    theta_oos = theta * z / np.sqrt(z ** 2 + u)
    return theta_is, theta_oos


def expected_oos_sharpe(n_assets: int, t_obs: int, theta: float) -> float:
    """E[θ̃_OOS] の閉形式（KWZ eq.12・₁F₁）。simulate との一致は tests で検証済み。"""
    n, t = n_assets, t_obs
    lg = (special.gammaln((n + 1) / 2.0) + special.gammaln((t - n + 2) / 2.0)
          + special.gammaln(t / 2.0) - special.gammaln((n + 2) / 2.0)
          - special.gammaln((t - n + 1) / 2.0) - special.gammaln((t + 1) / 2.0))
    return float(theta ** 2 * np.sqrt(t) / np.sqrt(2.0) * np.exp(lg)
                 * special.hyp1f1(0.5, (n + 2) / 2.0, -t * theta ** 2 / 2.0))


def oos_expected_shortfall(n_assets: int, t_obs: int, theta: float, *,
                           c: float = 0.5, n_sims: int = 200_000,
                           rng: np.random.Generator | None = None) -> float:
    """ES_c[θ̃] = E[θ̃ | θ̃ ≤ c 分位]（KWZ の break-even 定義に使う下側条件付き期待値）。"""
    _, oos = simulate_sharpes(n_assets, t_obs, theta, n_sims=n_sims, rng=rng)
    q = np.quantile(oos, c)
    return float(oos[oos <= q].mean())


def breakeven_theta(theta_baseline: float, n_assets: int = 2, t_obs: int = 120, *,
                    c: float = 0.5, n_sims: int = 200_000, tol: float = 1e-3) -> float:
    """break-even θ_b: 結合モデルの母集団 SR がこれを超えて初めて
    「推定リスク込みの OOS でベースライン θ₁ に勝つ」と言える水準（KWZ §4.1）。

    定義: ES_c[θ̃(θ_b)] = θ_baseline。c=0.5 が論文の基準ケース
    （Table 4: θ₁=0.10, T=120, N=2 → θ_b≈0.162＝+62% の増分が必要）。
    """
    rng = np.random.default_rng(12345)
    lo, hi = theta_baseline, max(theta_baseline * 5.0, theta_baseline + 1.0)
    if oos_expected_shortfall(n_assets, t_obs, hi, c=c, n_sims=n_sims,
                              rng=np.random.default_rng(12345)) < theta_baseline:
        raise ValueError("探索上限でも ES < θ₁（T が小さすぎる可能性）")
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        es = oos_expected_shortfall(n_assets, t_obs, mid, c=c, n_sims=n_sims,
                                    rng=np.random.default_rng(12345))
        if es < theta_baseline:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def grs_shifted_test(theta_hat_sq: float, theta_baseline_hat_sq: float,
                     n_assets: int, t_obs: int, delta_b_sq: float) -> tuple[float, float]:
    """シフト帰無の GRS 検定（KWZ eq.31/33）: H₀: δ² ≤ δ_b²（＝break-even 以下）。

    W = (T−N)(θ̂²−θ̂₁²)/[(N−1)(1+θ̂₁²)] ~ F_{N−1,T−N}(T·δ_b²/(1+θ̂₁²))。
    返り値 (W, p値)。p が小さい＝「増分は break-even を超える」証拠。
    """
    n, t = n_assets, t_obs
    w = (t - n) * (theta_hat_sq - theta_baseline_hat_sq) / (
        (n - 1) * (1.0 + theta_baseline_hat_sq))
    nc = t * delta_b_sq / (1.0 + theta_baseline_hat_sq)
    p = float(stats.ncf.sf(w, n - 1, t - n, nc)) if w > 0 else 1.0
    return float(w), p


def spanning_report(baseline_monthly_sr: float, combined_monthly_sr_hat: float,
                    baseline_monthly_sr_hat: float, t_obs: int, *,
                    n_assets: int = 2, c: float = 0.5) -> dict:
    """改良研究向けの一括レポート（表示・提案専用）。

    baseline_monthly_sr: 凍結ベースラインの母集団 SR の想定（保守的に）。
    *_hat: 標本値（月次）。返り値: θ_b・必要増分・E[θ̃]・GRS p 等。
    """
    theta_b = breakeven_theta(baseline_monthly_sr, n_assets, t_obs, c=c)
    delta_b_sq = theta_b ** 2 - baseline_monthly_sr ** 2
    w, p = grs_shifted_test(combined_monthly_sr_hat ** 2, baseline_monthly_sr_hat ** 2,
                            n_assets, t_obs, delta_b_sq)
    return {
        "theta_baseline": baseline_monthly_sr,
        "theta_breakeven": theta_b,
        "required_uplift": theta_b / baseline_monthly_sr - 1.0,
        "expected_oos_at_breakeven": expected_oos_sharpe(n_assets, t_obs, theta_b),
        "grs_W": w, "grs_p": p,
        "note": "表示・提案専用（DP18）。採用判断は人間ゲート。iid 正規仮定・"
                "ベースラインは凍結ウェイト前提（再推定なら適用外＝KWZ fn.10）",
    }
