"""判定器：任意戦略（の格子）を偽陽性排除メカニズム総動員で厳格に裁く。

中核思想＝「人が判定器をp-hackできないこと」：
- 全試行を TrialRegistry に scope 単位で事前登録（仮説＋経済的合理性が必須）。
- パラメータ格子の各点も独立した試行＝scope の K に算入。
- 各戦略の DSR は scope の K と Sharpe 分散でデフレート（試行を増やすほど基準が
  上がる＝「通るまで回す」が効かない）。
- 併せて PSR(真SR>0)・minTRL（認定に要する観測長）・サブ期間安定性・回転率・
  最大DD も報告し、PASS/FAIL を構造化レポートで返す。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..validation.dsr import (
    _moments, min_track_record_length, probabilistic_sharpe_ratio,
)
from ..validation.registry import TrialRegistry
from .engine import backtest


@dataclass
class StrategyVerdict:
    name: str
    params: dict
    n: int
    sr_ann: float
    psr: float
    dsr: float
    min_trl: float
    turnover: float
    max_dd: float
    hit: float
    sub: list = field(default_factory=list)   # [(label, ann_sharpe)]
    capacity_jpy: float = float("nan")        # 容量(¥)


@dataclass
class GridVerdict:
    scope: str
    k: int
    sr_var: float
    results: list           # StrategyVerdict, DSR降順
    best: object
    passed: bool
    report_md: str
    hypothesis: str = ""
    dsr_threshold: float = 0.95
    series: dict = field(default_factory=dict)   # name -> ネットリターン系列


def regime_breakdown(returns: pd.Series, regime: pd.Series,
                     ann: float = 252.0) -> pd.DataFrame:
    """戦略リターンをレジーム別に年率Sharpe・日数・平均に分解（ゲート判断の診断）。

    returns（決定日 index のネット系列）を ≤t の regime ラベルで groupby。**ゲートする前に**
    これを ungated 戦略で見る：MR の P&L が有利レジームに集中し不利レジームで負（Sharpe<0）
    なら、ゲートでサブ期間安定性が改善する見込み。分離が無ければゲートは無意味。regime は
    returns.index に reindex（ffill＝直近の確定レジーム）して整合。NaN ラベルは除外。
    """
    r = returns.dropna()
    lab = regime.reindex(r.index).ffill()
    rows = []
    for g, seg in r.groupby(lab):
        sd = seg.std(ddof=1)
        sh = (float(seg.mean() / sd * np.sqrt(ann))
              if len(seg) >= 2 and sd > 0 else float("nan"))
        rows.append((float(g), int(len(seg)), float(seg.mean()), sh))
    return pd.DataFrame(rows, columns=["regime", "n", "mean", "sharpe_ann"])


def _maxdd(r: pd.Series) -> float:
    cum = (1.0 + r).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _hit(r: pd.Series, npos: pd.Series) -> float:
    active = r[npos.reindex(r.index).fillna(0) > 0]
    return float((active > 0).mean()) if len(active) else float("nan")


def _subperiods(r: pd.Series, ann: float, k: int = 3) -> list:
    out = []
    for idx in np.array_split(np.arange(len(r)), k):
        s = r.iloc[idx]
        lbl = f"{s.index[0]:%Y-%m}..{s.index[-1]:%Y-%m}"
        sh = (s.mean() / s.std(ddof=1) * np.sqrt(ann)
              if len(s) >= 2 and s.std(ddof=1) > 0 else float("nan"))
        out.append((lbl, sh))
    return out


def _fmt_cap(x: float) -> str:
    """容量(¥)を読みやすく整形。"""
    if x is None or np.isnan(x):
        return "—"
    if x >= 1e8:
        return f"¥{x / 1e8:.0f}億"
    if x >= 1e4:
        return f"¥{x / 1e4:.0f}万"
    return f"¥{x:.0f}"


def judge_grid(strategies, view, *, scope: str, hypothesis: str,
               economic_rationale: str, registry: TrialRegistry,
               costs_bps: float = 15.0, price_field: str = "close",
               rebalance=None, dsr_threshold: float = 0.95,
               execution_lag: int = 0, adv=None, participation: float = 0.1,
               extra_trials: int = 0) -> GridVerdict:
    """戦略群（格子）を裁く。各点を事前登録＋記録し、scope の K でデフレート。

    execution_lag/adv/participation はバックテストの現実性（執行遅延・容量）に渡す。
    extra_trials: 探索しただけで建玉に至らない候補（CADF 等で事前棄却したペア）の数。
      K に算入し DSR をデフレートする（ペア探索の SBuMT 制御・DP13・KB §11.7）。
    """
    staged = []   # (strategy, result, returns, uuid)
    for s in strategies:
        res = backtest(s, view, costs_bps=costs_bps, price_field=price_field,
                       rebalance=rebalance, execution_lag=execution_lag,
                       adv=adv, participation=participation)
        r = res.returns.dropna()
        if r.size < 8 or r.std(ddof=1) == 0:
            continue
        sr, sk, ku, n = _moments(r.values)
        # 冪等記録：同一(scope,戦略,params)の再実行はKを水増ししない（永続運用向け）
        uid = registry.log_trial(scope=scope, strategy_id=s.name, params=s.params,
                                 sharpe=sr, n_obs=n, skew=sk, kurt=ku,
                                 hypothesis=hypothesis, rationale=economic_rationale)
        staged.append((s, res, r, uid))

    if extra_trials:                  # 探索しただけの候補も K に算入（DP13・KB §11.7）
        registry.log_scan_trials(scope=scope, count=int(extra_trials),
                                 hypothesis=hypothesis, rationale=economic_rationale)

    results = []
    for s, res, r, uid in staged:
        sr, sk, ku, n = _moments(r.values)
        dsr = registry.deflated_sharpe(uid)         # scope の K と V[SR] で自動デフレート
        psr = probabilistic_sharpe_ratio(sr, 0.0, n, sk, ku)
        try:
            mtrl = min_track_record_length(sr, 0.0, sk, ku, 0.95)
        except ValueError:
            mtrl = float("inf")
        results.append(StrategyVerdict(
            s.name, s.params, n, sr * np.sqrt(res.ann_factor), psr, dsr, mtrl,
            float(res.turnover.mean()), _maxdd(r), _hit(r, res.n_positions),
            _subperiods(r, res.ann_factor), res.capacity_jpy))

    results.sort(key=lambda v: v.dsr if not np.isnan(v.dsr) else -9, reverse=True)
    best = results[0] if results else None
    passed = bool(best and not np.isnan(best.dsr) and best.dsr >= dsr_threshold)
    k = registry.trial_count(scope)
    sr_var = registry.sharpe_variance(scope)
    report = _render(scope, k, sr_var, results, best, passed, hypothesis,
                     dsr_threshold)
    series = {s.name: r for s, res, r, uid in staged}
    return GridVerdict(scope, k, sr_var, results, best, passed, report,
                       hypothesis, dsr_threshold, series)


def _render(scope, k, sr_var, results, best, passed, hypothesis,
            thr) -> str:
    lines = [
        f"# 判定レポート: {scope}",
        f"- 仮説: {hypothesis}",
        f"- 試行数 K（この scope の累計）= **{k}**, 試行間SR分散 V[SR]={sr_var:.4f}",
        f"- 判定基準: DSR ≥ {thr}",
        "",
        "| strategy | SR(ann) | PSR(>0) | **DSR** | minTRL(月) | 回転 | maxDD | 容量 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for v in results:
        mtrl = "∞" if np.isinf(v.min_trl) else f"{v.min_trl:.0f}"
        lines.append(
            f"| {v.name} | {v.sr_ann:+.2f} | {v.psr:.2f} | **{v.dsr:.2f}** | "
            f"{mtrl} | {v.turnover:.2f} | {v.max_dd:.1%} | {_fmt_cap(v.capacity_jpy)} |")
    lines.append("")
    if passed:
        lines.append(f"## 判定: ✅ PASS — {best.name}（DSR={best.dsr:.3f} ≥ {thr}）")
        lines.append("多重検定後も有意。ただし実運用前に厳密OOS／容量／執行を要確認。")
    else:
        if best:
            seg = "  ".join(f"{l}:{s:+.2f}" for l, s in best.sub)
            lines.append(f"## 判定: ❌ FAIL — 最良 {best.name} でも "
                         f"DSR={best.dsr:.2f} < {thr}")
            lines.append(f"- 最良の内訳: SR(ann)={best.sr_ann:+.2f}, "
                         f"PSR(>0)={best.psr:.2f}, minTRL="
                         f"{'∞' if np.isinf(best.min_trl) else f'{best.min_trl:.0f}か月'}")
            lines.append(f"- サブ期間: {seg}")
        else:
            lines.append("## 判定: ❌ FAIL — 有効な戦略なし（データ/最小要件不足）")
        lines.append(f"- K={k} 試行に対しデフレート済み。**試行を増やすほど基準は上がる**"
                     "（＝判定器自体のp-hack不能）。")
    return "\n".join(lines)
