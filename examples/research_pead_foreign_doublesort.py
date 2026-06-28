r"""PEAD × 外国人持株：size 制御つき二重ソート / Fama-MacBeth / 判定（market-wide）。事前登録＝docs/51。

docs/50 の未解決点「foreign% は size の代理を超えるか（標本で反転）」を、市場全域所有（4,387社）の
breadth と size 直接制御で決着させる。
  A. 二重ソート（size3分位 × foreign3分位）で PEAD（SUE上位−下位 翌月）→ size各層で foreign 単調か。
  B. Fama-MacBeth: r₍t+1₎ = a + b·SUE + c·(SUE×foreign) + d·(SUE×size) + … → c<0 が d 制御後も有意か（決定的）。
  C. judge_grid（scope=pead_foreign_doublesort）で broad universe の条件付きPEADを大域デフレートDSR判定。

ユニバース＝top-1000 liquid common ∩ 所有 ∩ sue（ADV下限¥50M/日・PIT）。所有＝銘柄別 median foreign%。
実行: $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\research_pead_foreign_doublesort.py
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
from invest_system.equities.universe import apply_universe_mask, filter_common_stocks  # noqa: E402
from invest_system.equities.factors import cross_sectional_zscore, sector_neutralize  # noqa: E402
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.research import AsOfView, CrossSectionalStrategy, judge_grid, write_html  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import TrialRegistry, default_registry  # noqa: E402

TOPN = int(get_env("J_FDS_TOPN", "1000") or "1000")
ADV_FLOOR = float(get_env("J_FDS_ADV_FLOOR", "5e7") or "5e7")
COST_BPS = float(get_env("J_FDS_COST", "15") or "15")
OOS = "2024-01"
OWN = "data/edinet/ownership_categories.parquet"


def double_sort(sue_z, fwd, adv_me, fmed, uni, nq=3):
    """size3×foreign3 バケットの PEAD（SUE上位tercile−下位tercile 翌月平均, %/月）と FM t。"""
    buckets: dict = {(s, f): [] for s in range(nq) for f in range(nq)}
    for t in sue_z.index[:-1]:
        idx = uni.columns[uni.loc[t].to_numpy()]
        d = pd.DataFrame({"s": sue_z.loc[t, idx], "fw": fwd.loc[t, idx],
                          "adv": adv_me.loc[t, idx], "fc": fmed.reindex(idx)}).dropna()
        if len(d) < nq * nq * 6:
            continue
        st = pd.qcut(d["adv"].rank(method="first"), nq, labels=False)
        ft = pd.qcut(d["fc"].rank(method="first"), nq, labels=False)
        for si in range(nq):
            for fi in range(nq):
                b = d[(st == si) & (ft == fi)]
                if len(b) < 6:
                    continue
                hi = b[b["s"] >= b["s"].quantile(2 / 3)]["fw"]
                lo = b[b["s"] <= b["s"].quantile(1 / 3)]["fw"]
                if len(hi) >= 2 and len(lo) >= 2:
                    buckets[(si, fi)].append(hi.mean() - lo.mean())
    mat = pd.DataFrame(np.nan, index=[f"size{q}(small→large)"[:7] + str(q) for q in range(nq)],
                       columns=[f"frgn{q}" for q in range(nq)])
    tmat = mat.copy()
    for (si, fi), v in buckets.items():
        v = pd.Series(v)
        if len(v) > 2 and v.std(ddof=1) > 0:
            mat.iloc[si, fi] = v.mean() * 100
            tmat.iloc[si, fi] = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    return mat, tmat


def fama_macbeth(sue_z, fwd, adv_me, fmed, uni):
    """月次 OLS の係数時系列 → 平均・FM t・負割合。focus=SUE×foreign（c）。"""
    names = ["const", "SUE", "SUE*foreign", "SUE*size", "foreign", "size"]
    rows = []
    for t in sue_z.index[:-1]:
        idx = uni.columns[uni.loc[t].to_numpy()]
        d = pd.DataFrame({"s": sue_z.loc[t, idx], "fw": fwd.loc[t, idx],
                          "adv": adv_me.loc[t, idx], "fc": fmed.reindex(idx)}).dropna()
        if len(d) < 50:
            continue
        fr = d["fc"].rank(pct=True).to_numpy()
        ar = d["adv"].rank(pct=True).to_numpy()
        s = d["s"].to_numpy()
        X = np.column_stack([np.ones(len(d)), s, s * fr, s * ar, fr, ar])
        beta, *_ = np.linalg.lstsq(X, d["fw"].to_numpy(), rcond=None)
        rows.append(beta)
    B = np.asarray(rows)
    mean = B.mean(0)
    t = mean / (B.std(0, ddof=1) / np.sqrt(len(B)))
    return pd.DataFrame({"coef(×1e4)": mean * 1e4, "FM_t": t,
                         "%neg": (B < 0).mean(0) * 100}, index=names), len(B)


def main() -> int:
    if not Path(OWN).exists():
        print(f"ERROR: 所有パネル {OWN} がありません。build_ownership_panel.py を先に。")
        return 1
    sue = load_feature("sue_recent")
    adj = load_wide("adj_close"); turn = load_wide("turnover")
    close, high, low = load_wide("close"), load_wide("high"), load_wide("low")
    ul, ll, vol = load_wide("upper_limit"), load_wide("lower_limit"), load_wide("volume")
    me = sue.index
    listed = jq.fetch_listed_info().assign(Code=lambda d: d["Code"].astype(str))
    common = set(filter_common_stocks(listed)["Code"])
    sector = listed.set_index("Code")["S33"]
    own = pd.read_parquet(OWN); own["Code"] = own["Code"].astype(str)
    fmed = own.groupby("Code")["foreign_pct"].median()

    cols = [c for c in sue.columns if str(c) in common and str(c) in set(fmed.index)]
    adv_me = turn.rolling(60, min_periods=20).mean().reindex(me, method="ffill").reindex(columns=cols)
    rank_adv = adv_me.rank(axis=1, ascending=False)
    uni = (rank_adv <= TOPN) & (adv_me > ADV_FLOOR)
    fmed = fmed.reindex(cols)
    sue_z = cross_sectional_zscore(sector_neutralize(apply_universe_mask(sue.reindex(columns=cols), uni), sector))
    fwd = adj.reindex(me, method="ffill").reindex(columns=cols).pct_change().shift(-1)
    print(f"=== PEAD×外国人 二重ソート/FM/判定（docs/51）月末{len(me)} "
          f"{me.min():%Y-%m}..{me.max():%Y-%m}  uni日平均={uni.sum(axis=1).mean():.0f}（top{TOPN}）===")

    # A. 二重ソート
    mat, tmat = double_sort(sue_z, fwd, adv_me, fmed, uni)
    print("\n## A. 二重ソート PEAD（%/月・行=size昇順/列=foreign昇順 frgn0=低外国人）")
    print(mat.round(3).to_string())
    print("   (t値)\n" + tmat.round(2).to_string())
    print("   H1チェック（各 size 層で 低外国人frgn0 > 高外国人frgn2 か）:")
    for i in range(mat.shape[0]):
        print(f"     {mat.index[i]}: frgn0={mat.iloc[i,0]:+.3f} vs frgn2={mat.iloc[i,2]:+.3f}"
              f"  → {'OK(低>高)' if mat.iloc[i,0] > mat.iloc[i,2] else 'NG'}")

    # B. Fama-MacBeth
    fm, nm = fama_macbeth(sue_z, fwd, adv_me, fmed, uni)
    print(f"\n## B. Fama-MacBeth（{nm}ヶ月）＝決定的検定")
    print(fm.round(3).to_string())
    c = fm.loc["SUE*foreign"]
    print(f"   H2: SUE×foreign coef={c['coef(×1e4)']:+.2f}e-4, FM_t={c['FM_t']:+.2f}"
          f"  → {'foreign は size 超過で PEAD を弱める（c<0 有意）' if c['coef(×1e4)']<0 and abs(c['FM_t'])>=2 else 'size 超過の独立効果は非有意＝size代理の疑い'}")

    # C. 判定 L/S（broad universe）
    fcut = fmed.median(); acut = adv_me.median().median()
    lowF = set(fmed.index[fmed < fcut]); highF = set(fmed.index[fmed >= fcut])
    adv_stock = adv_me.median(); lowADV = set(adv_stock.index[adv_stock < acut])

    def mask(keep):
        m = sue_z.copy(); m[[c for c in cols if str(c) not in keep]] = np.nan; return m
    frank = fmed.rank(pct=True).reindex(cols).fillna(0.5)
    sig_xF = sue_z.mul(1.0 - frank, axis=1)
    strat = [CrossSectionalStrategy(sue_z, 0.2, name="pead_all"),
             CrossSectionalStrategy(mask(lowF), 0.2, name="pead_lowF"),
             CrossSectionalStrategy(mask(highF), 0.2, name="pead_highF"),
             CrossSectionalStrategy(mask(lowADV), 0.2, name="pead_lowADV"),
             CrossSectionalStrategy(sig_xF, 0.2, name="pead_xF")]
    nb, ns = (x.reindex(me).reindex(columns=cols) for x in
              limit_lock_flags(close, high, low, ul, ll, vol))
    view = AsOfView({"close": adj.reindex(me, method="ffill").reindex(columns=cols)})
    hyp = ("market-wide breadth と size 制御下で、外国人比率は size を超えて PEAD を弱める"
           "（低外国人で PEAD 大）。Jinushi2023 の所有チャネルの size 超過性を検定")
    rat = ("外国人=情報効率的な限界投資家ゆえ高外国人でドリフト消失。size と相関するが独立成分が"
           "あるかを二重ソート/FM/L-S で多面的に裁く")
    reg = TrialRegistry(get_env("J_FDS_REGISTRY")) if get_env("J_FDS_REGISTRY") else default_registry()
    with reg as rdb:
        v = judge_grid(strat, view, scope="pead_foreign_doublesort", hypothesis=hyp,
                       economic_rationale=rat, registry=rdb, costs_bps=COST_BPS,
                       execution_lag=0, adv=adv_me, no_buy=nb, no_sell=ns)
    print("\n## C. 判定 L/S\n" + v.report_md)
    write_html(v, "data/reports/pead_foreign_doublesort.html")
    print("--- net年率SR（broad universe）---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        oos = s[s.index >= pd.Timestamp(OOS)]
        print(f"  {r.name:<13} net={sharpe_ratio(s)*np.sqrt(12) if s.size>=8 else float('nan'):+.2f}"
              f" DSR={r.dsr:.2f} 容量={r.capacity_jpy/1e8:.2f}億"
              f" OOS={sharpe_ratio(oos)*np.sqrt(12) if oos.size>=8 else float('nan'):+.2f}")
    print("\n※ 2016+・top1000流動・所有=銘柄別median・size=ADV代理。判定=大域デフレートDSR。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
