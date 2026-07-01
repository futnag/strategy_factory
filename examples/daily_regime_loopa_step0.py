"""(a)ループ 段0：実 value/PEAD 一次の素エッジ（レジーム抜き・無フィルタ・コスト後）。

事前登録どおり、レジームを乗せる前に「実一次がそもそも正のエッジを持つか」を10銘柄で確認。
一次別に評価：エッジを持つ一次のみ段1（単層 vs 無フィルタ）へ・両落ちでループ打ち切り。
⚠ value 一次は「価格逆張り寄り」（分母 close 主導）と注記して解釈。PEAD は FY 正サプライズ後ドリフト（疎）。

読込 2020（value の 756日 trailing 確保）・評価 2024+（regime 推論窓と整合）・銘柄はロック済み10銘柄。
実行：python examples/daily_regime_loopa_step0.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from invest_system.equities.fundamentals import load_fundamentals
from invest_system.research.data_view import AsOfView
from invest_system.research.engine import backtest
from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime import fundamentals_pit as fp
from invest_system.research.daily_regime.cost import regime_cost_panel
from invest_system.research.daily_regime.strategy import PEADSurprisePrimary, ValueReversalPrimary
from invest_system.validation.dsr import sharpe_ratio

LOAD_START, EVAL_START = "2020-01-01", "2024-01-01"
LOCKED = [("23170", "5250", "T2"), ("135A0", "5250", "T3"), ("21240", "9050", "T2"),
          ("157A0", "9050", "T3"), ("26640", "6100", "T2"), ("26740", "6100", "T3"),
          ("27330", "6050", "T2"), ("26670", "6050", "T3"), ("65160", "3650", "T2"),
          ("38560", "3650", "T3")]


def psr(res):
    r = res.returns.dropna()
    try:
        return sharpe_ratio(r)
    except Exception:
        return np.nan


def main():
    t0 = time.perf_counter()
    codes = [c for c, _, _ in LOCKED]
    panels = drdata.load_daily_panels(start=LOAD_START, codes=codes)
    adj, va = panels.adj_close, panels.turnover
    fund = load_fundamentals(codes=codes)
    print(f"[load] adj {adj.shape} fund {fund.shape} ({time.perf_counter()-t0:.0f}s)", flush=True)

    eps_panel = fp.eps_asof_panel(fund, adj.index, lag_days=1)
    pead_panel = fp.pead_window_panel(fund, adj.index, D=20, lag_days=1)
    # PIT ゲート（恒久 leak_test と同型）：as-of EPS が開示+lag より前に出ていないこと
    fp.assert_fundamental_pit(eps_panel, fund, "EPS", lag_days=1)
    print(f"[pit] eps_asof PIT 確認 緑（DiscDate ラグ遵守）", flush=True)

    rows = []
    for code, sector, tier in LOCKED:
        if code not in adj.columns:
            continue
        ret = adj[code].pct_change(fill_method=None)
        amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(20).mean()
        cost = regime_cost_panel(pd.DataFrame({code: amh}), base_bps=10.0, impact_coef=20.0)
        frames = {"close": adj[[code]]}
        if code in eps_panel.columns:
            frames["eps_asof"] = eps_panel[[code]]
        if code in pead_panel.columns:
            frames["pead_window"] = pead_panel[[code]]
        view = AsOfView(frames)
        dates = view.dates[view.dates >= pd.Timestamp(EVAL_START)][:-2]
        n_pead = int(pead_panel[code].sum()) if code in pead_panel.columns else 0

        val = backtest(ValueReversalPrimary(code), view, costs_bps=cost, execution_lag=1, rebalance=dates)
        pead = (backtest(PEADSurprisePrimary(code), view, costs_bps=cost, execution_lag=1, rebalance=dates)
                if code in pead_panel.columns else None)
        rows.append({"code": code, "sec": sector, "tier": tier,
                     "value_sr": psr(val), "value_pos": int((val.returns != 0).sum()),
                     "pead_sr": psr(pead) if pead is not None else np.nan, "pead_days": n_pead})
        print(f"  {code}({sector}/{tier}): value_sr={psr(val):+.3f} "
              f"pead_sr={(psr(pead) if pead is not None else float('nan')):+.3f} pead_days={n_pead}", flush=True)

    df = pd.DataFrame(rows)
    print("\n===== 段0：一次単体の素エッジ（無フィルタ・コスト後 per-period Sharpe・10銘柄） =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    n = len(df)
    for col, label in [("value_sr", "value 一次（⚠価格逆張り寄り）"), ("pead_sr", "PEAD 一次（FY正サプライズ後ドリフト・疎）")]:
        s = df[col].dropna()
        pos = int((s > 0).sum())
        med = float(s.median()) if len(s) else np.nan
        qualifies = (len(s) > 0) and (pos / len(s) >= 2 / 3) and (med > 0)
        print(f"\n{label}: 正 {pos}/{len(s)} ({pos/max(len(s),1):.0%})・中央値 {med:+.3f} → "
              f"段1進出資格={'YES' if qualifies else 'NO'}（基準 ≥2/3 かつ 中央値>0）", flush=True)
    print("\n[note] value=価格逆張り寄り（ファンダ割安でなく短期リバーサル）。PEAD=FY年1回で疎＝符号傾向止まり。", flush=True)
    print(f"[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
