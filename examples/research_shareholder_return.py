"""仮説検証①：株主還元イベント（配当予想サプライズ・自社株買い実績）【柱C 拡張】。

経済的根拠（Chan の問い）：配当・自社株買いは経営者の私的情報を伴う**コミットメント・
シグナル**（カットの社会的コストが高いため、増配は持続的な収益力への自信の表明）。
2023年の東証「資本コストと株価を意識した経営」（PBR改革）以降、株主還元の強化は
日本株最大の構造レジームで、還元変化の漸進的織り込みが歪みとして残るという仮説。
反対側＝還元の変化を月次リバランスでしか織り込まない指数資金・低感度機関。

シグナル（PIT・fins_summary 全件ミラー）：
  div_revision … 予想年間配当（FDivAnn）の開示順改訂率。**株式分割は AdjFactor 累積で
                 補正**（2:1分割の機械的減配を中立化＝events.dividend_forecast_revision）
  buyback      … 自己株比率（TrShFY/ShOutFY）の開示順差分＝**実現した**自社株買い。
                 比率なので分割に不変（events.buyback_intensity）
設計上の要点：配当改訂は決算と同時開示が多く**既存 PEAD（利益予想改訂）と交絡**する。
直交化版（クロスセクションで pead に回帰した残差）を主形として判定し、生版との差で
「配当固有の情報」を識別する。ショート脚は §7.2 の教訓（減配銘柄ショート＝value の
ロングと衝突）により持たない＝全戦略ロングティルト（上位ロング×ユニバース等加重ヘッジ）。

事前登録（K=4）：divrev_lt / divrev_orth_lt（主形）/ buyback_lt / sr_combo_lt（直交版50/50）。
判定は §6.9-6.11 と同一の現実性（月次・15bps・容量）。OOS 2024-01〜＋PBR改革前後
（2023-04 分割）を併記。

実行: .venv\\Scripts\\python.exe examples\\research_shareholder_return.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.data.store import load_wide  # noqa: E402
from invest_system.equities import events  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CompositeStrategy, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS, REFORM = "2016-07", "2026-05", "2024-01", "2023-04"
SCOPE = "shareholder_return"


def _sr(x: pd.Series, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


def _orth(za: pd.DataFrame, zb: pd.DataFrame) -> pd.DataFrame:
    """各月クロスセクションで za を zb に回帰した残差を再標準化（直交化）。"""
    out = {}
    for t in za.index:
        a = za.loc[t]
        b = zb.loc[t] if t in zb.index else pd.Series(dtype="float64")
        pair = pd.concat({"a": a, "b": b}, axis=1).dropna()
        if len(pair) < 10 or pair["b"].var() == 0:
            out[t] = a
            continue
        beta = pair["a"].cov(pair["b"]) / pair["b"].var()
        resid = a - beta * b.reindex(a.index)
        sd = resid.std()
        out[t] = (resid - resid.mean()) / sd if sd and sd > 0 else resid
    return pd.DataFrame(out).T.reindex(za.index)


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    fund = load_fundamentals(superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))

    # --- シグナル構築（PIT・分割調整）---
    af = load_wide("adj_factor")
    adj_cum = af.where(af.notna(), 1.0).cumprod() if not af.empty else None
    divrev_l = events.dividend_forecast_revision(fund, adj_cum=adj_cum)
    bb_l = events.buyback_intensity(fund)
    pead_l = events.forecast_revision(fund)
    divrev = zN(point_in_time(divrev_l, rebal, ["div_revision"], date_col="DiscDate",
                              lag_days=1)["div_revision"].reindex(columns=superset))
    buyback = zN(point_in_time(bb_l, rebal, ["buyback"], date_col="DiscDate",
                               lag_days=1)["buyback"].reindex(columns=superset))
    pead = zN(point_in_time(pead_l, rebal, ["fcst_revision"], date_col="DiscDate",
                            lag_days=1)["fcst_revision"].reindex(columns=superset))
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    divrev_o = _orth(divrev, pead)

    # --- 独立性の診断（建玉前）：既存シグナルとの平均クロスセクション相関 ---
    def xcorr(a, b):
        cs = [a.loc[t].corr(b.loc[t]) for t in a.index
              if t in b.index and a.loc[t].notna().sum() > 30]
        return float(np.nanmean(cs)) if cs else float("nan")
    print(f"=== 株主還元イベント（{START}〜{END}・月末{len(rebal)}回・scope={SCOPE}）===")
    print(f"独立性（平均XS相関）: div_rev×pead {xcorr(divrev, pead):+.2f} / "
          f"div_rev×value {xcorr(divrev, value):+.2f} / "
          f"buyback×value {xcorr(buyback, value):+.2f} / "
          f"div_rev_orth×pead {xcorr(divrev_o, pead):+.2f}")

    q = 0.2
    s_div = CrossSectionalStrategy(divrev, q, name="divrev_lt", long_only=True)
    s_divo = CrossSectionalStrategy(divrev_o, q, name="divrev_orth_lt", long_only=True)
    s_bb = CrossSectionalStrategy(buyback, q, name="buyback_lt", long_only=True)
    s_cmb = CompositeStrategy([s_divo, s_bb], [0.5, 0.5], name="sr_combo_lt")

    with default_registry() as reg:
        v = judge_grid(
            [s_div, s_divo, s_bb, s_cmb], view, scope=SCOPE,
            hypothesis=("配当予想の上方改訂と実現自社株買いは経営者のコミットメント・シグナルで、"
                        "発表後も漸進的に織り込まれる（PEAD の還元版）。PBR改革（2023-04〜）で"
                        "構造的追い風。利益改訂（PEAD）と直交な成分にも情報があるか"),
            economic_rationale=("減配・買い戻し中止の社会的コストが高い日本では、還元強化は持続的"
                                "収益力への自信の表明＝私的情報を含む。反対側は還元変化に低感度の"
                                "指数資金。分割は AdjFactor で補正済み・ショート脚は §7.2 の教訓"
                                "により持たない（ロングティルトのみ）。"),
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）・PBR改革前後（{REFORM} 分割）---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        print(f"  {r.name:<16} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"改革前={_sr(s, hi=REFORM):+.2f} 改革後={_sr(s, lo=REFORM):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f}")
    print("\n※ 直交版（divrev_orth_lt）が生版と同等以上なら「配当固有の情報」が存在する"
          "証拠。改革後だけ効く場合はレジーム製品＝フォワード依存と正直に解釈する。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
