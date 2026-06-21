"""仮説検証：マイクロキャップ（低流動性帯）の1ヶ月クロスセクション逆張り。

ブリーフィング文書 §2.2 の中心主張＝「日本市場はリバーサルが支配的・特に TSE Growth など
個人主体でアナリストカバーの薄いマイクロキャップで顕著」を、ファクトリの厳格規律で裁く。
既存の reversal 検証（§6.7 price_factors_gold・reversal_5）は**流動性上位500**＝大中型で
FAIL だったが、文書はエッジを**機関が入れない低流動性帯**に置く。本スクリプトはその土俵を
直接踏む：

- ユニバース＝PIT 低流動性帯（trailing 12m 売買代金で上位300より下の [301,1000]）＝機関レーダー外。
  別変種で Growth 市場(Mkt=0113)のみも検証。生存者バイアス排除（各時点 ≤t の流動性のみ）。
- シグナル＝rev = −(月次リターン) を S33 セクター中立・XS z 化（決定 t→翌月実現＝PIT）。
- 現実性＝値幅制限ロック(DP15 no_buy/no_sell)＋容量(participation×ADV)を常設。
- コスト＝判定は片道30bps（マイクロキャップとして楽観的な床）で scope 登録。別途コスト感応
  スイープ 15→200bps（throwaway・K 不変）と T+1 始値執行で損益分岐を露出。

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_microcap_reversal.py
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
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, universe_members,
)
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize,
)
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import AsOfView, CrossSectionalStrategy, judge_grid, write_html  # noqa: E402
from invest_system.research.engine import backtest, open_fill_backtest  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import TrialRegistry, default_registry  # noqa: E402

OOS = get_env("J_MR_OOS", "2024-01") or "2024-01"
REG_PATH = get_env("J_MR_REGISTRY", None)
BAND_LO = int(get_env("J_MR_BAND_LO", "300") or "300")   # 上位300より下＝機関レーダー外
BAND_HI = int(get_env("J_MR_BAND_HI", "1000") or "1000")
COST_BPS = float(get_env("J_MR_COST", "30") or "30")     # 判定コスト（片道・楽観的な床）


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ser = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(ser.groupby(idx.to_period("M")).max().values)


def band_universe(turn_me: pd.DataFrame, keep: set[str], lo: int, hi: int,
                  lookback: int = 12, min_obs: int = 6) -> pd.DataFrame:
    """PIT 流動性帯マスク：各月末に trailing 中央値売買代金で降順ランクし [lo,hi) を True。

    lo=300,hi=1000 なら「上位300より下〜1000位＝低流動性マイクロキャップ帯」。各時点 ≤t の
    流動性のみ使用（先読み・生存者バイアス無し）。t 時点で実取引中（turnover 非NaN）に限定。
    keep は対象にする Code 集合（普通株 or Growth など）。
    """
    cols = [c for c in turn_me.columns if str(c) in keep]
    sub = turn_me[cols]
    mask = pd.DataFrame(False, index=sub.index, columns=sub.columns)
    for i, t in enumerate(sub.index):
        window = sub.iloc[max(0, i - lookback + 1): i + 1]
        obs = window.notna().sum(axis=0)
        med = window.median(axis=0, skipna=True)
        trading = sub.loc[t].notna()
        eligible = med[(obs >= min_obs) & trading].sort_values(ascending=False)
        band = eligible.index[lo:hi]
        mask.loc[t, band] = True
    return mask


def growth_universe(turn_me: pd.DataFrame, keep: set[str], min_obs: int = 6,
                    lookback: int = 12) -> pd.DataFrame:
    """Growth 市場の全 eligible 銘柄（PIT・流動性帯で絞らず＝TSE Growth そのもの）。"""
    cols = [c for c in turn_me.columns if str(c) in keep]
    sub = turn_me[cols]
    mask = pd.DataFrame(False, index=sub.index, columns=sub.columns)
    for i, t in enumerate(sub.index):
        window = sub.iloc[max(0, i - lookback + 1): i + 1]
        obs = window.notna().sum(axis=0)
        trading = sub.loc[t].notna()
        elig = obs[(obs >= min_obs) & trading].index
        mask.loc[t, elig] = True
    return mask


def main() -> int:
    adj = load_wide("adj_close")
    adj_open = load_wide("adj_open")
    turn = load_wide("turnover")
    close = load_wide("close")
    high = load_wide("high")
    low = load_wide("low")
    ul = load_wide("upper_limit")
    ll = load_wide("lower_limit")
    vol = load_wide("volume")
    if adj.empty or turn.empty:
        print("ERROR: Silver 未生成。store.materialize_all() を先に実行してください。")
        return 1
    me = _month_ends(adj.index)
    adj_me, turn_me = adj.loc[me], turn.loc[me]
    print(f"=== マイクロキャップ逆張り検証  月末{len(me)}  帯[{BAND_LO},{BAND_HI})  コスト{COST_BPS:.0f}bps ===")

    listed = jq.fetch_listed_info()
    listed = listed.assign(Code=listed["Code"].astype(str))
    common = set(filter_common_stocks(listed)["Code"])
    growth = set(listed[listed["Mkt"].astype(str) == "0113"]["Code"]) & common
    sector = listed.set_index("Code")["S33"]

    # 値幅制限ロック（DP15）：日次で判定し月末に整列（執行バー=月末終値）
    no_buy_d, no_sell_d = limit_lock_flags(close, high, low, ul, ll, vol)
    no_buy, no_sell = no_buy_d.loc[me], no_sell_d.loc[me]

    ret_1m = adj_me.pct_change()                # 各月末で既知の過去1ヶ月リターン（PIT）
    rev_raw = -ret_1m                           # 逆張り＝負けをロング

    universes = {
        "microcap_band": band_universe(turn_me, common, BAND_LO, BAND_HI),
        "growth_all": growth_universe(turn_me, growth),
    }

    def build_signal(mask: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
        superset = universe_members(mask)
        m = mask.reindex(index=me, columns=superset).fillna(False)
        rev = cross_sectional_zscore(sector_neutralize(
            apply_universe_mask(rev_raw.reindex(index=me, columns=superset), m), sector))
        adv = turn_me.reindex(index=me, columns=superset)
        return rev, superset, adv

    rev_mc, sup_mc, adv_mc = build_signal(universes["microcap_band"])
    rev_gr, sup_gr, adv_gr = build_signal(universes["growth_all"])
    print(f"ユニバース  microcap_band: superset={len(sup_mc)} 月平均={universes['microcap_band'].sum(axis=1).mean():.0f}")
    print(f"            growth_all   : superset={len(sup_gr)} 月平均={universes['growth_all'].sum(axis=1).mean():.0f}")

    view_mc = AsOfView({"close": adj_me.reindex(columns=sup_mc)})
    view_gr = AsOfView({"close": adj_me.reindex(columns=sup_gr)})

    strategies = [
        CrossSectionalStrategy(rev_mc, 0.2, name="rev_ls_q20"),
        CrossSectionalStrategy(rev_mc, 0.1, name="rev_ls_q10"),
        CrossSectionalStrategy(rev_mc, 0.2, name="rev_longonly_q20", long_only=True),
        CrossSectionalStrategy(rev_gr, 0.2, name="rev_growth_ls_q20"),
    ]

    # 判定：microcap_band 系は view_mc、growth 系は view_gr。judge_grid は単一 view 前提のため
    # growth 戦略はシグナルに growth superset を持たせ、view も growth superset で別建てにする。
    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    hyp = ("1ヶ月クロスセクション逆張りは、機関が入れない低流動性マイクロキャップ/Growth で"
           "こそ耐久・コスト生存エッジを持つ（ブリーフィング文書 §2.2 の中心主張）")
    rat = ("短期リバーサル＝流動性供給/オーバーリアクションのプレミアム。文書は個人主体・カバー"
           "希薄な低流動性帯で顕著と主張。値幅制限・容量・往復スプレッドを織り込んで真偽を裁く")
    with reg_cm as reg_db:
        v_mc = judge_grid(strategies[:3], view_mc, scope="microcap_reversal",
                          hypothesis=hyp, economic_rationale=rat, registry=reg_db,
                          costs_bps=COST_BPS, adv=adv_mc, no_buy=no_buy, no_sell=no_sell)
        v_gr = judge_grid([strategies[3]], view_gr, scope="microcap_reversal",
                          hypothesis=hyp, economic_rationale=rat, registry=reg_db,
                          costs_bps=COST_BPS, adv=adv_gr, no_buy=no_buy, no_sell=no_sell)
    print("\n[microcap_band]\n" + v_mc.report_md)
    print("\n[growth_all]\n" + v_gr.report_md)
    write_html(v_mc, "data/reports/microcap_reversal.html")

    # IS/OOS・前後2020・gross 対比
    print(f"\n--- IS/OOS（保留 {OOS}〜・年率Sharpe）・gross 対比 ---")
    for v, view, adv in [(v_mc, view_mc, adv_mc), (v_gr, view_gr, adv_gr)]:
        for r in v.results:
            strat = next(s for s in strategies if s.name == r.name)
            res = backtest(strat, view, costs_bps=COST_BPS, adv=adv,
                           no_buy=no_buy, no_sell=no_sell)
            ls = res.returns.dropna()
            gross = res.gross.dropna()
            is_ = ls[ls.index < pd.Timestamp(OOS)]
            oos = ls[ls.index >= pd.Timestamp(OOS)]
            si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
            so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
            gsr = sharpe_ratio(gross) * np.sqrt(12) if gross.size >= 8 else np.nan
            (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
            print(f"  {r.name:<20} net全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
                  f"gross={gsr:+.2f} | IS={si:+.2f} OOS={so:+.2f} | "
                  f"前2020={pre:+.2f} 後={post:+.2f} | 容量={r.capacity_jpy/1e8:.2f}億 "
                  f"blocked計={int(res.n_blocked.sum())}")

    # コスト感応スイープ（throwaway・K 不変）＝損益分岐スプレッドの露出
    print("\n--- コスト感応（rev_ls_q20・microcap_band・片道bps→net年率SR）---")
    base = strategies[0]
    for c in [0, 15, 30, 50, 75, 100, 150, 200]:
        res = backtest(base, view_mc, costs_bps=float(c), adv=adv_mc,
                       no_buy=no_buy, no_sell=no_sell)
        sr = sharpe_ratio(res.returns.dropna()) * np.sqrt(12)
        print(f"  {c:4d}bps  net年率SR={sr:+.2f}  回転={res.turnover.mean():.2f}")

    # T+1 始値執行（同足バイアス排除・最も保守的）
    print("\n--- T+1 始値執行（rev_ls_q20・microcap_band・30bps）---")
    wbd = {t: base.target_weights(view_mc.asof(t)) for t in view_mc.dates}
    wbd = {t: w for t, w in wbd.items() if len(w)}
    res_o = open_fill_backtest(wbd, adj_open.reindex(columns=sup_mc),
                               costs_bps=COST_BPS, name="rev_ls_q20_T+1open")
    lo = res_o.returns.dropna()
    print(f"  net全SR={sharpe_ratio(lo)*np.sqrt(12):+.2f}  "
          f"gross={sharpe_ratio(res_o.gross.dropna())*np.sqrt(12):+.2f}  n={lo.size}")
    print("\n  ※ 文書の主張通りなら net がコスト後も正に残るはず。"
          "残らなければ『マイクロキャップ逆張り＝スプレッドで死ぬ紙上アルファ』を実証。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
