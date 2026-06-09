"""仮説検証：Regime-Switching Mean Reversion（柱D・Ernie Chan 補完・KB §11）。

§6.4/§6.6 の共和分ペア平均回帰は最良でも DSR 0.03（FAIL）。失敗要因は（①多重検定 K、
②サブ期間の符号反転）。本研究は **②をレジームスイッチで改善できるか** を正直に検証する。

設計：
- 同業種・形成窓CADFで選別したペアを**等加重プール**（CompositeStrategy）に束ねた baseline。
- そのプールを RegimeGated で覆った変種（市場トレンド強度／ボラのレジームで建玉をゲート）。
- baseline と全変種を**同一 scope** で judge（K を公正に共有）＋スキャン多重を extra_trials 計上。
- ゲート前に regime_breakdown で「P&L がどのレジームに集中するか」を診断（分離が無ければ無意味）。

PIT：regime[t] は close[t] 由来＝z[t] と同 as-of、執行は execution_lag=1（翌足＝同足先読み無）。
レジーム閾値は拡張窓 percentile（≤t 分布）＝先読み無。レジーム定義は経済的合理性から事前固定
（多数試して最良を採らない＝判定器の p-hack 不能・KB §11.7）。

プールの日次ウェイトは一度だけ PIT 生成して _Replay で再利用（共和分再検定の重複を回避）。
取得済み日足ミラー（data/jquants/daily/）で動作・API 不要。環境変数で規模/レジーム調整可：
  J_MRG_M(業種内上位) J_MRG_FORM_DAYS J_MRG_LOOKBACK J_MRG_ENTRY J_MRG_CADF_P J_MRG_METHOD
  J_MRG_ER_WIN(ER窓) J_MRG_VOL_WIN J_MRG_MINP(拡張窓最小) J_MRG_START J_MRG_SCOPE J_MRG_REGISTRY
実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_meanrev_regime.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:                                  # Windows コンソール(cp932)でも日本語/記号を出力
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                     # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities.panel import load_daily_panel  # noqa: E402
from invest_system.equities.universe import filter_common_stocks  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CointegratedPairs, CompositeStrategy, RegimeGated, Strategy,
    judge_grid, regime_breakdown, write_html,
)
from invest_system.timeseries import cadf, trend_regime, vol_regime  # noqa: E402
from invest_system.validation.registry import (  # noqa: E402
    TrialRegistry, default_registry,
)

M = int(get_env("J_MRG_M", "4") or "4")                       # 業種内・流動性上位M
FORM_DAYS = int(get_env("J_MRG_FORM_DAYS", "504") or "504")   # 形成窓(IS)≈2年
LOOKBACK = int(get_env("J_MRG_LOOKBACK", "60") or "60")
ENTRY = float(get_env("J_MRG_ENTRY", "2.0") or "2.0")
CADF_P = float(get_env("J_MRG_CADF_P", "0.05") or "0.05")
METHOD = get_env("J_MRG_METHOD", "rolling_ols") or "rolling_ols"
ER_WIN = int(get_env("J_MRG_ER_WIN", "60") or "60")           # Efficiency Ratio 窓
VOL_WIN = int(get_env("J_MRG_VOL_WIN", "60") or "60")
MINP = int(get_env("J_MRG_MINP", "252") or "252")             # 拡張窓三分位の最小観測
START = get_env("J_MRG_START", None)
SCOPE = get_env("J_MRG_SCOPE", "meanrev_regime") or "meanrev_regime"
REG_PATH = get_env("J_MRG_REGISTRY", None)


class _Replay(Strategy):
    """事前計算した日次ウェイト（PIT生成済み）を date 引きで返す（再計算回避）。"""

    def __init__(self, weights: dict, name: str, params: dict):
        self._w = weights
        self.name = name
        self.params = params

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _coverage(strat: Strategy, view: AsOfView, dates) -> float:
    """建玉率＝ウェイトが非空な決定日の割合（ゲートの過剰な手仕舞いを検出）。"""
    return float(np.mean([not strat.target_weights(view.asof(t)).empty
                          for t in dates]))


def main() -> int:
    print(f"=== Regime-Switching MR（柱D）  上位{M}・形成窓{FORM_DAYS}・{METHOD}"
          f"・ER窓{ER_WIN}・scope={SCOPE} ===")
    px = load_daily_panel(field="AdjC", start=START)
    if px.empty:
        print("ERROR: 日足ミラー(data/jquants/daily/)が空。download_jquants.py を先に実行。")
        return 1
    adv_full = load_daily_panel(field="Va", start=START)
    print(f"日足パネル: {px.shape[0]}日 × {px.shape[1]}銘柄")
    if px.shape[0] <= FORM_DAYS + LOOKBACK + 10:
        print("ERROR: 形成窓に対して履歴が短すぎます（J_MRG_FORM_DAYS / J_MRG_START を調整）。")
        return 1

    listed = jq.fetch_listed_info()
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    sec = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]

    form = px.iloc[:FORM_DAYS]                          # 形成窓(IS)＝取引窓に先行
    trade_dates = px.index[FORM_DAYS:]

    liq = adv_full.iloc[:FORM_DAYS].median()           # 形成窓ADV中央値で流動性上位
    sub = sec[sec.index.isin(px.columns) & sec.index.isin(common)].dropna()
    candidates: list[tuple[str, str]] = []
    for _, grp in sub.groupby(sub):
        codes = liq.reindex([c for c in grp.index if c in px.columns]).dropna()
        top = codes.sort_values(ascending=False).head(M).index.tolist()
        candidates += list(combinations(top, 2))
    n_scanned = len(candidates)
    print(f"候補ペア（業種内・上位{M}・{sub.nunique()}業種）: {n_scanned}")

    tradeable: list[tuple[str, str]] = []              # 形成窓CADFで事前選別（IS＝先読み無）
    for a, b in candidates:
        join = pd.concat([form[a], form[b]], axis=1).dropna()
        if len(join) < max(60, LOOKBACK + 5):
            continue
        _, p = cadf(join.iloc[:, 0], join.iloc[:, 1])
        if np.isfinite(p) and p <= CADF_P:
            tradeable.append((a, b))
    print(f"共和分ペア（形成窓 CADF p<={CADF_P}）: {len(tradeable)} / {n_scanned}")
    if not tradeable:
        print("形成窓で共和分ペア無し。J_MRG_M / J_MRG_FORM_DAYS / J_MRG_CADF_P を調整。")
        return 0

    view = AsOfView({"close": px})
    adv = adv_full.reindex(index=trade_dates)

    # --- プールの日次ウェイトを一度だけ PIT 生成（共和分再検定の重複を回避）---
    pairs = [CointegratedPairs(a, b, lookback=LOOKBACK, entry=ENTRY, method=METHOD,
                               coint_gate=True, cadf_max_p=CADF_P)
             for a, b in tradeable]
    pool = CompositeStrategy(pairs, name="meanrev_pool")        # 等加重（1/N）プール
    print(f"プール構築（{len(pairs)}ペア）→ {len(trade_dates)}日のウェイト生成中…")
    wmap = {t: pool.target_weights(view.asof(t)) for t in trade_dates}
    pool_params = {"n_pairs": len(pairs), "lookback": LOOKBACK, "entry": ENTRY,
                   "method": METHOD, "pool": "equal_weight"}
    base = _Replay(wmap, name="meanrev_pool", params=pool_params)

    # --- レジーム系列（市場・拡張窓三分位・PIT）。定義は事前固定 ---
    trend = trend_regime(px, window=ER_WIN, min_periods=MINP)   # 0=レンジ…2=強トレンド
    vol = vol_regime(px, window=VOL_WIN, min_periods=MINP)      # 0=低…2=高ボラ
    strategies = [
        base,
        RegimeGated(base, trend, allowed={0, 1},
                    name="meanrev_pool|trend<=1(no_strong_trend)"),
        RegimeGated(base, trend, allowed={0},
                    name="meanrev_pool|range_only(trend=0)"),
        RegimeGated(base, vol, allowed={0, 1},
                    name="meanrev_pool|vol<=1(no_high_vol)"),
    ]

    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("共和分ペア平均回帰のサブ期間不安定（符号反転）は、強トレンド/"
                        "高ボラのレジームを避ければ改善し DSR が上がるか（Regime-Switching）"),
            economic_rationale=("強トレンド・急変局面では共和分が崩壊しスプレッドがトレンド化"
                                "（相関崩壊）。レンジ/平常ボラ局面に建玉を限定すると相対価値の"
                                "回帰が効きやすい。レジームは経済的に動機づけ・事前固定。"),
            registry=reg, costs_bps=15.0, execution_lag=1, rebalance=trade_dates,
            adv=adv, extra_trials=n_scanned - len(tradeable))
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # --- 診断：プール baseline の P&L をレジーム別に分解（ゲートの当否を見る）---
    bs = v.series.get("meanrev_pool", pd.Series(dtype="float64"))
    print("\n--- regime_breakdown（baseline プール・年率Sharpe）---")
    print("[トレンド強度 0=レンジ/1=中/2=強トレンド]")
    print(regime_breakdown(bs, trend).to_string(index=False,
          float_format=lambda x: f"{x:+.3f}"))
    print("[ボラ 0=低/1=中/2=高]")
    print(regime_breakdown(bs, vol).to_string(index=False,
          float_format=lambda x: f"{x:+.3f}"))

    print("\n--- 建玉率（time-in-market）---")
    for s in strategies:
        print(f"  {s.name:<42} {_coverage(s, view, trade_dates):.1%}")
    print(f"\nK={v.k}（プール変種{len(strategies)} ＋ スキャン{n_scanned - len(tradeable)}）"
          f"。レジームは事前固定＝定義探索で K を水増ししない（KB §11.7・DP13）。")
    print("※ 判断：有利レジームで Sharpe>>不利（できれば不利<0）＝分離があり、かつ gated の "
          "DSR>baseline かつサブ期間符号が安定して初めてレジームに価値。建玉率の過小も確認。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
