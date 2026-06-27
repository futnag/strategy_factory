r"""仮説検証 候補①テキストα Stage 1：有報の前年比テキスト変化（Lazy Prices型）。

docs/45 の事前登録の Stage 1（**新規依存ゼロ・完全PIT**）。有報MD&Aの前年比コサイン類似度
（文字n-gram, sklearn）が低い＝記述を書き換えた銘柄は将来アンダーパフォーム——を判定器で裁く。
合格＝DSR≥0.95 ＋ value/momentum 残差化後も残る ＋ 小型で効き大型で消える（H2）＋ サブ期間符号一貫。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_text_lazy_prices.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import disclosure_text as dtx  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe, universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import fundamentals_panel, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_residualize, cross_sectional_zscore, sector_neutralize,
    value_quality_size_factors, winsorize_cross_sectional,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N = 500
QS = [0.1, 0.2, 0.3]
IDX_CACHE = Path("data/processed/edinet_annual_index.parquet")
TXT_CACHE = Path("data/processed/disclosure_text.parquet")
DOCS = Path("data/edinet/docs")
FIELDS = ["ShOutFY", "TrShFY", "Eq", "FEPS", "FNP", "FOP", "FSales", "FDivAnn",
          "CFO", "NP", "TA"]


def _avg_xs_corr(a, b, min_names=20):
    vals = []
    for t in a.index:
        x, y = a.loc[t], b.reindex(columns=a.columns).loc[t]
        m = x.notna() & y.notna()
        if int(m.sum()) >= min_names and x[m].std() > 0 and y[m].std() > 0:
            vals.append(float(np.corrcoef(x[m].to_numpy(float), y[m].to_numpy(float))[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def _ls_sharpe(sig, fwd, mask, q=0.2, min_names=15):
    """mask 内で signal 上位/下位 q を EW ロングショート → 年率Sharpe（局在診断）。"""
    out = []
    for t in sig.index:
        s = sig.loc[t].where(mask.loc[t]).dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(len(s) * q))
        o = s.sort_values()
        r = fwd.loc[t]
        rl, rs = r.reindex(o.index[-k:]).mean(), r.reindex(o.index[:k]).mean()
        if pd.notna(rl) and pd.notna(rs):
            out.append(rl - rs)
    x = pd.Series(out)
    return float(x.mean() / x.std() * np.sqrt(12)) if len(x) >= 8 and x.std() > 0 else float("nan")


def load_text(superset) -> pd.DataFrame:
    """有報PIT索引→本文抽出（キャッシュ・冪等再開）。返り値 [Code, submitDateTime, text]。"""
    idx = (pd.read_parquet(IDX_CACHE) if IDX_CACHE.exists()
           else dtx.annual_report_index())
    IDX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not IDX_CACHE.exists():
        idx.to_parquet(IDX_CACHE)
    idx = idx[idx["Code"].isin(superset)]
    cached = (pd.read_parquet(TXT_CACHE) if TXT_CACHE.exists()
              else pd.DataFrame(columns=["Code", "submitDateTime", "docID", "text"]))
    have = set(cached["docID"]) if len(cached) else set()
    todo = idx[~idx["docID"].isin(have)]
    print(f"有報索引（ユニバース）{len(idx)} / 抽出済 {len(have)} / 新規抽出 {len(todo)}")
    new = []
    for i, (_, r) in enumerate(todo.iterrows()):
        zp = DOCS / f"{r['docID']}_5.zip"
        txt = dtx.extract_narrative(zp) if zp.exists() else ""
        new.append({"Code": r["Code"], "submitDateTime": r["submitDateTime"],
                    "docID": r["docID"], "text": txt})
        if (i + 1) % 1500 == 0:
            cached = pd.concat([cached, pd.DataFrame(new)], ignore_index=True)
            new = []
            cached.to_parquet(TXT_CACHE)
            print(f"  抽出 {i + 1}/{len(todo)} …")
    if new:
        cached = pd.concat([cached, pd.DataFrame(new)], ignore_index=True)
        cached.to_parquet(TXT_CACHE)
    t = cached[cached["Code"].isin(superset)].copy()
    t = t[t["text"].astype(str).str.len() > 200]
    return t[["Code", "submitDateTime", "text"]]


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    print(f"=== ①テキストα Stage1（Lazy Prices） {START}〜{END} 上位{TOP_N} ===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    print(f"パネル {adj.shape[0]}ヶ月 × ユニバース{len(superset)}銘柄")

    texts = load_text(set(superset))
    print(f"テキスト保有有報 {len(texts)} 件 / {texts['Code'].nunique()} 銘柄")
    sim = dtx.yoy_text_similarity(texts).dropna(subset=["text_sim"])
    print(f"前年比類似度 {len(sim)} 件（中央値 {sim['text_sim'].median():.3f}）")
    pit_t = point_in_time(sim, rebal, ["text_sim"], date_col="submitDateTime",
                          code_col="Code", lag_days=1)
    ts = pit_t["text_sim"].reindex(columns=superset)
    cov = int((~ts.isna()).sum().sum())
    print(f"text_sim as-of 有効セル {cov:,}")

    def prep(f):
        fm = apply_universe_mask(f.reindex(columns=superset), umask)
        return cross_sectional_zscore(sector_neutralize(
            winsorize_cross_sectional(fm), sector))

    sig = prep(ts)                                  # 高類似度(記述安定)=ロング
    pit_f = fundamentals_panel(rebal, FIELDS, codes=superset, lag_days=1)
    vqs = value_quality_size_factors(pit_f, raw, adj)
    val, mom = prep(vqs["book_to_market"]), prep(vqs["momentum"])
    sz = prep(vqs["size"])
    resid = cross_sectional_zscore(cross_sectional_residualize(sig, [val, mom]))

    strats = [CrossSectionalStrategy(sig, q, name=f"text_sim(q={q})") for q in QS]
    strats.append(CrossSectionalStrategy(resid, 0.2, name="text_sim_resid(q=0.2)"))

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="disclosure_text_change",
            hypothesis="有報MD&Aの前年比テキスト類似度が低い（記述を書き換えた）銘柄は将来"
                       "アンダーパフォームする（Lazy Prices／soft information のアンダーリアクション）",
            economic_rationale="経営者の叙述には数値に出ない soft information があり、不利な実態を"
                               "言葉で薄める/物語を書き換える行為は注意の限界で緩やかに織り込まれる。"
                               "価格/ファンダの単調変換では作れない独立軸。反対側は叙述を読まない参加者。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1, execution_lag=1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print("\n--- 独立性（text_sim と既知factorの月平均XS相関）---")
    for nm, o in (("value(B/M)", val), ("momentum", mom), ("size", sz)):
        print(f"  vs {nm:<12} ρ̄ = {_avg_xs_corr(sig, o):+.2f}")

    # H2 局在：小型 vs 大型で L/S Sharpe を対比
    shares = (pit_f["ShOutFY"].reindex(columns=superset)
              .sub(pit_f["TrShFY"].reindex(columns=superset), fill_value=0.0))
    mcap = (raw * shares.where(shares > 0)).reindex(columns=superset)
    large = mcap.gt(mcap.median(axis=1), axis=0).reindex(columns=superset).fillna(False)
    small = (mcap.notna() & ~large)
    fwd = adj.pct_change().shift(-1)
    print("\n--- H2 局在（小型で効き大型で消えるか・q=0.2 L/S 年率Sharpe）---")
    print(f"  小型半分 SR={_ls_sharpe(sig, fwd, small):+.2f} / "
          f"大型半分 SR={_ls_sharpe(sig, fwd, large):+.2f}（小型>大型 が仮説）")

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
