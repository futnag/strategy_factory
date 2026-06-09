"""仮説検証：同業種・共和分ペアの平均回帰（柱D・Ernie Chan 補完・KB §11）。

§6.4 の素朴 pairs(DSR 0.02) を、共和分ゲート＋AsOf 動的ヘッジ（CointegratedPairs）に昇格。
各業種(S33)内の流動性上位 M 銘柄で全ペアを候補化し、形成窓(IS)の CADF で事前選別 → 取引窓
(OOS)で judge_grid 判定。ペア探索の SBuMT は extra_trials で全候補を K に算入する（DP13）。

PIT：形成窓は取引窓に先行（先読みなし）。建玉は coint_gate=True で各 t の直近窓を再検定し
共和分の崩壊に追随。執行は execution_lag=1（z は当日終値で算出→翌日執行＝同足先読み排除）。

取得済み日足ミラー（data/jquants/daily/）で動作・API 不要。環境変数で規模/変種調整可：
  J_MR_M(業種内上位数) J_MR_FORM_DAYS(形成窓) J_MR_LOOKBACK J_MR_ENTRY J_MR_CADF_P
  J_MR_METHOD(rolling_ols/kalman) J_MR_MAX_HL(半減期上限日) J_MR_COINT_GATE(0で無効)
  J_MR_SCOPE(試行scope名) J_MR_START(AdjC開始日) J_MR_REGISTRY(台帳パス・未指定で永続)
実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_meanrev_pairs.py
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
from invest_system.equities.universe import filter_common_stocks  # noqa: E402
from invest_system.equities.panel import load_daily_panel  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CointegratedPairs, judge_grid, write_html,
)
from invest_system.timeseries import cadf  # noqa: E402
from invest_system.validation.registry import TrialRegistry, default_registry  # noqa: E402

M = int(get_env("J_MR_M", "5") or "5")                  # 業種内・流動性上位M銘柄
FORM_DAYS = int(get_env("J_MR_FORM_DAYS", "504") or "504")  # 形成窓(IS)日数≈2年
LOOKBACK = int(get_env("J_MR_LOOKBACK", "60") or "60")
ENTRY = float(get_env("J_MR_ENTRY", "2.0") or "2.0")
CADF_P = float(get_env("J_MR_CADF_P", "0.05") or "0.05")
START = get_env("J_MR_START", None)                     # AdjC開始日（未指定=全期間）
REG_PATH = get_env("J_MR_REGISTRY", None)               # 未指定=永続レジストリ
METHOD = get_env("J_MR_METHOD", "rolling_ols") or "rolling_ols"  # rolling_ols / kalman
_mhl = get_env("J_MR_MAX_HL", None)
MAX_HL = float(_mhl) if _mhl else None                  # 半減期上限(日)・任意ゲート
COINT_GATE = (get_env("J_MR_COINT_GATE", "1") or "1") != "0"     # 各tのCADF再検定
SCOPE = get_env("J_MR_SCOPE", "meanrev_pairs") or "meanrev_pairs"


def main() -> int:
    print(f"=== 共和分ペア平均回帰（柱D）  上位{M}・形成窓{FORM_DAYS}日・{METHOD}"
          f"・gate={COINT_GATE}・scope={SCOPE} ===")
    px = load_daily_panel(field="AdjC", start=START)
    if px.empty:
        print("ERROR: 日足ミラー(data/jquants/daily/)が空。download_jquants.py を先に実行。")
        return 1
    adv_full = load_daily_panel(field="Va", start=START)
    print(f"日足パネル: {px.shape[0]}日 × {px.shape[1]}銘柄")
    if px.shape[0] <= FORM_DAYS + LOOKBACK + 10:
        print("ERROR: 形成窓に対して履歴が短すぎます（J_MR_FORM_DAYS / J_MR_START を調整）。")
        return 1

    listed = jq.fetch_listed_info()
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    sec = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]

    # 形成窓(IS) / 取引窓(OOS) に分割（形成は取引に先行＝先読み無）
    form = px.iloc[:FORM_DAYS]
    trade_dates = px.index[FORM_DAYS:]

    # 業種内・流動性(形成窓のADV中央値)上位M → 全ペア候補
    liq = adv_full.iloc[:FORM_DAYS].median()
    sub = sec[sec.index.isin(px.columns) & sec.index.isin(common)].dropna()
    candidates: list[tuple[str, str]] = []
    for _, grp in sub.groupby(sub):
        codes = liq.reindex([c for c in grp.index if c in px.columns]).dropna()
        top = codes.sort_values(ascending=False).head(M).index.tolist()
        candidates += list(combinations(top, 2))
    n_scanned = len(candidates)
    print(f"候補ペア（業種内・上位{M}・{sub.nunique()}業種）: {n_scanned}")

    # 形成窓 CADF で事前選別（IS のみ＝先読み無）
    tradeable: list[tuple[str, str]] = []
    for a, b in candidates:
        join = pd.concat([form[a], form[b]], axis=1).dropna()
        if len(join) < max(60, LOOKBACK + 5):
            continue
        _, p = cadf(join.iloc[:, 0], join.iloc[:, 1])
        if np.isfinite(p) and p <= CADF_P:
            tradeable.append((a, b))
    print(f"共和分ペア（形成窓 CADF p<={CADF_P}）: {len(tradeable)} / {n_scanned}")
    if not tradeable:
        print("形成窓で共和分ペア無し。J_MR_M / J_MR_FORM_DAYS / J_MR_CADF_P を調整。")
        return 0

    strategies = [CointegratedPairs(a, b, lookback=LOOKBACK, entry=ENTRY,
                                    method=METHOD, coint_gate=COINT_GATE,
                                    cadf_max_p=CADF_P, max_half_life=MAX_HL)
                  for a, b in tradeable]
    view = AsOfView({"close": px})
    adv = adv_full.reindex(index=trade_dates)

    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis="同業種・共和分ペアの一時乖離は平均回帰する（相対価値）",
            economic_rationale=("同業種は共通ファンダで結ばれ、在庫/流動性ショックによる"
                                "一時的な価格乖離が回帰する。共和分が崩れた局面は建玉しない。"),
            registry=reg, costs_bps=15.0, execution_lag=1, rebalance=trade_dates,
            adv=adv, extra_trials=n_scanned - len(tradeable))
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))
    print(f"\nK={v.k}（建玉候補{len(tradeable)} ＋ スキャン{n_scanned - len(tradeable)}）"
          f"＝ペア探索の多重検定を全候補で計上（DP13・KB §11.7）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
