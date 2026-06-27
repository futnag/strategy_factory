"""因果構造の structural break 検知。

ローリング PCMCI で VALUE→RET エッジ強度を追跡し、CUSUM + ruptures PELT と
統計的レジーム検知（HMM）を比較する。パフォーマンスブレイク（平均リターン変化）
との先行性も報告。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .graph import CausalGraphResult, learn_pcmci_graph


@dataclass
class StructuralBreakResult:
    """1セクターの構造ブレイク検知結果。"""
    s33: str
    value_edge_strength: pd.Series          # ローリング VALUE→RET 係数/MCI
    causal_alarms: list[pd.Timestamp]       # 因果構造 CUSUM 警報
    perf_alarms: list[pd.Timestamp]         # パフォーマンス CUSUM 警報
    ruptures_alarms: list[pd.Timestamp]     # ruptures PELT
    hmm_regimes: Optional[pd.Series] = None
    edge_diff_events: list[pd.Timestamp] = field(default_factory=list)


def page_cusum(x: pd.Series, k_sd: float = 0.5, h_sd: float = 4.0,
               min_periods: int = 60) -> list[pd.Timestamp]:
    """因果的 Page CUSUM（≤t 拡張窓基準）。"""
    mu = x.expanding(min_periods=min_periods).mean().shift(1)
    sd = x.expanding(min_periods=min_periods).std().shift(1)
    s_hi = s_lo = 0.0
    alarms = []
    for t, v in x.items():
        m, d = mu.get(t), sd.get(t)
        if not (np.isfinite(m) and np.isfinite(d) and np.isfinite(v) and d > 0):
            continue
        z = (v - m) / d
        s_hi = max(0.0, s_hi + z - k_sd)
        s_lo = min(0.0, s_lo + z + k_sd)
        if s_hi > h_sd or s_lo < -h_sd:
            alarms.append(pd.Timestamp(t))
            s_hi = s_lo = 0.0
    return alarms


def _partial_regression_coef(y: pd.Series, x: pd.Series,
                             controls: pd.DataFrame) -> float:
    """y ~ x + controls の x 係数（backdoor 調整済み因果効果の近似）。"""
    df = pd.concat([y.rename("y"), x.rename("x"), controls], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    Z = df.drop(columns=["y", "x"]).values
    A = np.column_stack([np.ones(len(df)), df["x"].values, Z])
    b, *_ = np.linalg.lstsq(A, df["y"].values, rcond=None)
    return float(b[1])


def rolling_value_edge(panel: pd.DataFrame, window: int = 252, step: int = 21,
                       adjustment_vars: Optional[list[str]] = None) -> pd.Series:
    """ローリング窓で VALUE→RET の調整済み因果効果（偏回帰係数）を推定。

    全窓 PCMCI 再学習は計算コスト過大のため、全期間 PCMCI で得た調整集合を
    固定し、窓ごとに backdoor 偏回帰でエッジ強度を追跡する（因果的監視の実用近似）。
    """
    if adjustment_vars is None:
        adjustment_vars = [c for c in panel.columns
                           if c not in ("VALUE", "RET")][:4]
    adj = panel[[v for v in adjustment_vars if v in panel.columns]]
    y = panel["RET"].shift(-1)
    x = panel["VALUE"]
    idx = panel.dropna(subset=["VALUE", "RET"]).index
    out = pd.Series(np.nan, index=idx)
    for i in range(window, len(idx) + 1, step):
        sl = idx[i - window:i]          # 窓末 = idx[i-1] = t
        sl_fit = sl[:-1]                # RET[sl[-1]+1] は t 時点未実現のため除外
        coef = _partial_regression_coef(
            y.loc[sl_fit], x.loc[sl_fit], adj.loc[sl_fit],
        )
        out.loc[idx[i - 1]] = coef
    return out.ffill()


def _ruptures_breaks(series: pd.Series, pen: float = 5.0) -> list[pd.Timestamp]:
    """ruptures PELT で変化点検知。"""
    try:
        import ruptures as rpt
    except ImportError:
        return []
    y = series.dropna().values.reshape(-1, 1)
    if len(y) < 60:
        return []
    algo = rpt.Pelt(model="rbf", min_size=30, jump=5).fit(y)
    bk = algo.predict(pen=pen)
    idx = series.dropna().index
    return [pd.Timestamp(idx[b - 1]) for b in bk if b < len(idx)]


def _hmm_regimes(series: pd.Series, n_states: int = 2) -> pd.Series:
    """Gaussian HMM でレジームラベル（統計的ベースライン）。"""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return pd.Series(dtype=float)
    y = series.dropna().values.reshape(-1, 1)
    if len(y) < 100:
        return pd.Series(dtype=float)
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=100, random_state=42)
    model.fit(y)
    states = model.predict(y)
    return pd.Series(states, index=series.dropna().index)


def detect_structural_breaks(panel: pd.DataFrame, s33: str, *,
                             window: int = 252, step: int = 21,
                             adjustment_vars: Optional[list[str]] = None,
                             skip_graph_diff: bool = False) -> StructuralBreakResult:
    """因果構造ブレイクとパフォーマンスブレイクを同時監視。"""
    edge = rolling_value_edge(panel, window=window, step=step,
                              adjustment_vars=adjustment_vars)
    perf = panel["RET"].rolling(window).mean()

    causal_alarms = page_cusum(edge.diff().dropna(), h_sd=5.0)
    perf_alarms = page_cusum(perf.diff().dropna(), h_sd=5.0)
    rpt_alarms = _ruptures_breaks(edge)
    hmm = _hmm_regimes(edge)

    # グラフ構造変化：連続窓のエッジ集合 Jaccard 距離が大きい日
    edge_events = ([] if skip_graph_diff
                   else _graph_structure_events(panel, window=window, step=step * 3))

    return StructuralBreakResult(
        s33=s33, value_edge_strength=edge,
        causal_alarms=causal_alarms, perf_alarms=perf_alarms,
        ruptures_alarms=rpt_alarms, hmm_regimes=hmm,
        edge_diff_events=edge_events,
    )


def _graph_structure_events(panel: pd.DataFrame, window: int, step: int,
                            tau_max: int = 3, threshold: float = 0.5) -> list[pd.Timestamp]:
    """連続ローリング窓の親集合 Jaccard 距離 > threshold の日。"""
    idx = panel.dropna().index
    prev_parents: Optional[set[str]] = None
    events = []
    for i in range(window, len(idx) + 1, step):
        sl = idx[i - window:i]
        sub = panel.loc[sl].dropna()
        if len(sub) < window // 2:
            continue
        try:
            res = learn_pcmci_graph(sub, tau_max=tau_max, min_obs=window // 3)
            parents = {f"{p[0]}_L{p[1]}" for plist in res.parents.values()
                       for p in plist if p[1] >= 1}
            if prev_parents is not None:
                inter = len(parents & prev_parents)
                union = len(parents | prev_parents) or 1
                jaccard = inter / union
                if jaccard < (1.0 - threshold):
                    events.append(pd.Timestamp(idx[i - 1]))
            prev_parents = parents
        except Exception:  # noqa: BLE001
            continue
    return events


def compare_detection_methods(result: StructuralBreakResult, *,
                              lead_days: int = 60) -> pd.DataFrame:
    """因果 vs 統計的 vs パフォーマンス検知の比較表。"""
    rows = []
    perf_set = set(result.perf_alarms)
    for label, alarms in [
        ("causal_cusum", result.causal_alarms),
        ("ruptures", result.ruptures_alarms),
        ("graph_diff", result.edge_diff_events),
    ]:
        lead = sim = lag = iso = 0
        for p in sorted(perf_set):
            near = [a for a in alarms
                    if abs((pd.Timestamp(a) - pd.Timestamp(p)).days) <= lead_days]
            if not near:
                iso += 1
                continue
            d = min(near, key=lambda a: abs((pd.Timestamp(a) - pd.Timestamp(p)).days))
            gap = (pd.Timestamp(d) - pd.Timestamp(p)).days
            if gap < -10:
                lead += 1
            elif gap > 10:
                lag += 1
            else:
                sim += 1
        rows.append({
            "method": label, "n_alarms": len(alarms),
            "lead": lead, "simultaneous": sim, "lag": lag, "isolated_perf": iso,
        })
    return pd.DataFrame(rows)