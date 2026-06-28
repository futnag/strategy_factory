r"""所有構造で条件付けた PEAD（value+PEAD 旗艦の精緻化）の一回判定。事前登録＝docs/50。

Jinushi(2023)＝PEAD は低外国人/高個人に集中。本スクリプトは旗艦の PEAD シグナル（sue_recent）を
foreign%/individual% で条件付け（低外国人へ限定/傾斜）し、素 PEAD を上回るかを **大域デフレートDSR・
コスト・容量・値幅ロック** で公正に裁く。最重要ガード＝**foreign が size の代理を超えるか**（lowADV 対照）。

ユニバース＝流動 top-300 普通株 ∩ EDINET-DB 所有取得済み（~160・`data/edinet/ownership_categories.parquet`）。
シグナル＝`load_feature("sue_recent")`（本決算 surprise yield・月末PIT）をセクター中立 z 化。
所有/size 分類＝各銘柄の期間 median（持続タイプ）で median 2分割。

実行: $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\research_pead_ownership_tilt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.data.store import load_wide  # noqa: E402
from invest_system.data.feature_store import load_feature  # noqa: E402
from invest_system.equities.universe import filter_common_stocks  # noqa: E402
from invest_system.equities.factors import cross_sectional_zscore, sector_neutralize  # noqa: E402
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import AsOfView, CrossSectionalStrategy, judge_grid, write_html  # noqa: E402
from invest_system.research.engine import backtest, open_fill_backtest  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import TrialRegistry, default_registry  # noqa: E402

OOS = get_env("J_POT_OOS", "2024-01") or "2024-01"
REG_PATH = get_env("J_POT_REGISTRY", None)
COST_BPS = float(get_env("J_POT_COST", "15") or "15")
Q = float(get_env("J_POT_Q", "0.2") or "0.2")
OWN = "data/edinet/ownership_categories.parquet"


def _ann_sr(r, ann=12.0):
    r = pd.Series(r).dropna()
    return sharpe_ratio(r) * np.sqrt(ann) if r.size >= 8 else float("nan")


def main() -> int:
    if not Path(OWN).exists():
        print(f"ERROR: 所有パネル {OWN} がありません。先に取得してください。")
        return 1
    sue = load_feature("sue_recent")                 # 月末 Date×Code（PIT surprise yield）
    adj = load_wide("adj_close"); adj_open = load_wide("adj_open")
    turn = load_wide("turnover")
    close, high, low = load_wide("close"), load_wide("high"), load_wide("low")
    ul, ll, vol = load_wide("upper_limit"), load_wide("lower_limit"), load_wide("volume")
    me = sue.index                                   # 月末
    print(f"=== 所有条件付きPEAD 判定（事前登録 docs/50）月末{len(me)} {me.min():%Y-%m}..{me.max():%Y-%m} ===")

    listed = jq.fetch_listed_info().assign(Code=lambda d: d["Code"].astype(str))
    common = set(filter_common_stocks(listed)["Code"])
    sector = listed.set_index("Code")["S33"]

    own = pd.read_parquet(OWN); own["Code"] = own["Code"].astype(str)
    med = own.groupby("Code")[["foreign_pct", "individual_pct"]].median()
    uni = [c for c in sue.columns if str(c) in set(med.index) and str(c) in common]
    med = med.loc[[c for c in med.index if c in uni]]
    print(f"ユニバース（top300流動 ∩ 所有取得 ∩ 普通株）= {len(uni)} 銘柄")

    # trailing ADV（容量＋size対照）。per-stock median で size 分類
    adv_me = turn.rolling(60, min_periods=20).mean().reindex(me).reindex(columns=uni)
    adv_stock = adv_me.median()
    # 持続タイプの median 2分割
    fmed, imed, amed = med["foreign_pct"].median(), med["individual_pct"].median(), adv_stock.median()
    lowF = set(med.index[med["foreign_pct"] < fmed])
    highF = set(med.index[med["foreign_pct"] >= fmed])
    highI = set(med.index[med["individual_pct"] >= imed])
    lowADV = set(adv_stock.index[adv_stock < amed])
    print(f"分割: lowF={len(lowF)} highF={len(highF)} highIndiv={len(highI)} lowADV={len(lowADV)}"
          f"  (foreign med={fmed:.1f}%, indiv med={imed:.1f}%)")
    # foreign と size の重なり（H2 が意味を持つか）
    overlap = len(lowF & lowADV) / max(len(lowF), 1)
    print(f"lowF ∩ lowADV / lowF = {overlap:.2f}（1.00 なら foreign=size 完全一致＝H2 検定不能）")

    # セクター中立 z シグナル（PIT）
    sig = cross_sectional_zscore(sector_neutralize(sue.reindex(columns=uni), sector))

    def mask(cols: set) -> pd.DataFrame:
        keep = [c for c in uni if c in cols]
        m = sig.copy()
        drop = [c for c in uni if c not in cols]
        m[drop] = np.nan
        return m

    # 連続傾斜：低外国人ほど加重（foreign_rank∈[0,1]→ weight=1−rank）
    frank = med["foreign_pct"].rank(pct=True)
    w_lowF = (1.0 - frank).reindex(uni).fillna(0.5)
    sig_xF = sig.mul(w_lowF, axis=1)

    strategies = [
        CrossSectionalStrategy(sig, Q, name="pead_all"),
        CrossSectionalStrategy(mask(lowF), Q, name="pead_lowF"),
        CrossSectionalStrategy(mask(highF), Q, name="pead_highF"),
        CrossSectionalStrategy(mask(highI), Q, name="pead_highIndiv"),
        CrossSectionalStrategy(mask(lowADV), Q, name="pead_lowADV"),
        CrossSectionalStrategy(sig_xF, Q, name="pead_xF"),
    ]

    no_buy_d, no_sell_d = limit_lock_flags(close, high, low, ul, ll, vol)
    no_buy, no_sell = no_buy_d.reindex(me), no_sell_d.reindex(me)
    view = AsOfView({"close": adj.reindex(me).reindex(columns=uni)})

    hyp = ("PEAD を低外国人/高個人サブユニバースに条件付けると素PEADを上回る（Jinushi2023＝PEADは"
           "低外国人/高個人に集中）。foreign は size の代理を超えてPEADを予測する")
    rat = ("外国人=情報効率的な限界投資家。低外国人/高個人=underreaction が持続。旗艦の流動top300は"
           "高外国人中心ゆえPEADが弱い→低外国人へ傾けると drift を捕捉。size超過分を lowADV 対照で検定")
    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg_db:
        v = judge_grid(strategies, view, scope="pead_ownership_tilt",
                       hypothesis=hyp, economic_rationale=rat, registry=reg_db,
                       costs_bps=COST_BPS, execution_lag=0, adv=adv_me,
                       no_buy=no_buy, no_sell=no_sell)
    print("\n" + v.report_md)
    write_html(v, "data/reports/pead_ownership_tilt.html")

    # 仮説の直接対比（throwaway 診断・judge の系列を再利用）
    def sr_of(name):
        s = v.series.get(name)
        return _ann_sr(s) if s is not None else float("nan")
    print("\n--- 仮説対比（net 年率SR）---")
    print(f"  H1: pead_lowF {sr_of('pead_lowF'):+.2f}  vs  pead_all {sr_of('pead_all'):+.2f}"
          f"  / pead_highIndiv {sr_of('pead_highIndiv'):+.2f}  / pead_xF {sr_of('pead_xF'):+.2f}")
    print(f"  対照: pead_highF {sr_of('pead_highF'):+.2f}（弱いはず）")
    print(f"  H2(size超過): pead_lowF {sr_of('pead_lowF'):+.2f}  vs  pead_lowADV {sr_of('pead_lowADV'):+.2f}"
          f"  → lowF>lowADV なら foreign は size を超える")

    # IS/OOS・前後2020・T+1始値（lowF と all）
    print(f"\n--- IS/OOS(保留{OOS})・前後2020・T+1始値 ---")
    for s in strategies:
        res = backtest(s, view, costs_bps=COST_BPS, execution_lag=0, adv=adv_me,
                       no_buy=no_buy, no_sell=no_sell)
        net = res.returns.dropna()
        is_ = net[net.index < pd.Timestamp(OOS)]; oos = net[net.index >= pd.Timestamp(OOS)]
        (_, pre), (_, post) = pre_post_sharpe(net, "2020-01-01")
        wbd = {t: s.target_weights(view.asof(t)) for t in view.dates}
        wbd = {t: w for t, w in wbd.items() if len(w)}
        ro = open_fill_backtest(wbd, adj_open.reindex(columns=uni), costs_bps=COST_BPS).returns.dropna()
        print(f"  {s.name:<15} net={_ann_sr(net):+.2f} | IS={_ann_sr(is_):+.2f} OOS={_ann_sr(oos):+.2f}"
              f" | 前2020={pre:+.2f} 後={post:+.2f} | T+1={_ann_sr(ro):+.2f}")
    print(f"\n※ 期間2016+・{len(uni)}銘柄（top300流動 ∩ 所有取得）・所有はFY2019+の銘柄別median。"
          "判定=scope=pead_ownership_tilt の大域デフレートDSR。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
