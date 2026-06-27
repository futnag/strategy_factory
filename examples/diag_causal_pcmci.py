"""外生ドライバ→因子の時系列因果図学習（PCMCI+）と因果効果の構造ブレイク監視。

docs/41 の昇格パス（§5）の実装。docs/41 は**価格由来**の構造（βmkt・偏相関）が
性能ブレイクに先行しないことを示した。本診断は **価格に外生なドライバ**（VIX・
USDJPY・SP500・金利・コモディティ）を使い、

  (1) PCMCI+（tigramite）で「ドライバ→因子」の時系列因果図を学習し、
      正しい調整集合（adjustment set）＝各因子の因果的親を同定する。
  (2) 学習した因果リンク（ドライバ[t-L]→因子[t]）の**因果効果**をローリングで推定し、
      その**構造ブレイク**を CUSUM で監視。これが因子の**パフォーマンスブレイク**に
      先行するか（docs/41 で価格由来は先行しなかった）を検証する。

すべて data/ のローカルキャッシュのみ（API キー不要・オフライン）。
判定（DSR/PASS-FAIL）は一切しない。現象記述の診断であり戦略ではない。

> docs/41 の §5 で「日次の外生ドライバが無い」としたのは誤り：
> data/supplemental/macro_extended.parquet 等に**日次**ドライバが揃っている。
> 本診断はそれを使う（docs/41 §5 の制約は解消）。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_causal_daily import _resid, ls_return, page_cusum, _winsorize  # noqa: E402

DATA = ROOT / "data"
FEAT = DATA / "features"
SUP = DATA / "supplemental"

TRUE_BREAKS = ["2018-10", "2020-03", "2020-12", "2022-09"]


# ---------------------------------------------------------------- ドライバ構築
def build_drivers() -> pd.DataFrame:
    """日次・外生ドライバを定常化して返す（金利→差分bp・価格→対数リターン・VIX→変化）。"""
    m = pd.read_parquet(SUP / "macro_extended.parquet").sort_index()
    m = m[~m.index.duplicated(keep="last")]
    d = pd.DataFrame(index=m.index)
    d["VIX_chg"] = m["vix"].diff()
    d["USDJPY_ret"] = np.log(m["usd_jpy"]).diff()
    d["SP500_ret"] = np.log(m["sp500"]).diff()
    d["JP10Y_chg"] = m["japan_10y_yield"].diff()        # %pt
    d["US10Y_chg"] = m["us_10y_yield"].diff()
    d["WTI_ret"] = np.log(m["wti_crude"]).diff()
    d["COPPER_ret"] = np.log(m["copper"]).diff()
    return d.replace([np.inf, -np.inf], np.nan)


def build_factors() -> pd.DataFrame:
    ret = _winsorize(pd.read_parquet(FEAT / "returns.parquet"))
    mom = pd.read_parquet(FEAT / "momentum_12_1.parquet")
    rev = pd.read_parquet(FEAT / "reversal_5.parquet")
    vol = pd.read_parquet(FEAT / "vol_20.parquet")
    f = pd.DataFrame({
        "MOM":    ls_return(mom, ret, sign=+1.0),
        "REV":    ls_return(rev, ret, sign=-1.0),
        "LOWVOL": ls_return(vol, ret, sign=-1.0),
    })
    return f


# -------------------------------------------------------- ローリング多変量係数
def rolling_causal_effect(y: pd.Series, X: pd.DataFrame, target: str,
                          w: int = 252) -> pd.Series:
    """y ~ [const + X] のローリング回帰で、target 列の係数（因果効果）を返す。
    X は調整集合（因果的親をそのラグでシフト済みの列）。≤窓のみ（因果）。"""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    cols = list(X.columns)
    ti = cols.index(target)
    yv = df["y"].values
    Xv = df[cols].values
    idx = df.index
    out = pd.Series(np.nan, index=idx)
    for e in range(w, len(idx) + 1):
        s = e - w
        A = np.column_stack([np.ones(w), Xv[s:e]])
        b, *_ = np.linalg.lstsq(A, yv[s:e], rcond=None)
        out.iloc[e - 1] = b[1 + ti]
    return out


def _fmt(ds, n=8):
    ds = list(ds)
    s = "  ".join(f"{pd.Timestamp(d):%Y-%m}" for d in ds[:n])
    return (s + (f"  …(+{len(ds)-n})" if len(ds) > n else "")) or "—"


# ===========================================================================
def main() -> int:
    warnings.filterwarnings("ignore")
    print("=" * 78)
    print("外生ドライバ→因子の因果図学習(PCMCI+)＋因果効果の構造ブレイク監視")
    print("（K不変・判定なし・オフライン）")
    print("=" * 78)

    fac = build_factors()
    drv = build_drivers()
    factors = ["MOM", "REV", "LOWVOL"]
    drivers = list(drv.columns)

    # 共通営業日で整合（因子LSの実現日 d に、ドライバの当日値・ラグを合わせる）
    panel = pd.concat([fac[factors], drv[drivers]], axis=1).dropna()
    print(f"整合パネル {panel.index.min():%Y-%m-%d}〜{panel.index.max():%Y-%m-%d}  "
          f"{len(panel)}日 / 変数 {panel.shape[1]}（因子{len(factors)}＋ドライバ{len(drivers)}）")

    # ---- 標準化して PCMCI+ ------------------------------------------------
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    allvars = factors + drivers
    Z = (panel[allvars] - panel[allvars].mean()) / panel[allvars].std()
    dataframe = pp.DataFrame(Z.values, var_names=allvars)
    tau_max = 5
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    print(f"\nPCMCI+ 実行（ParCorr・tau_max={tau_max}・pc_alpha=0.01）…")
    res = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=0.01)
    val, graph = res["val_matrix"], res["graph"]
    name = {i: v for i, v in enumerate(allvars)}

    # ---- Part 1: 各因子の因果的親（ドライバ→因子）------------------------
    print("\n" + "=" * 78)
    print("Part 1  学習された因果図：各因子の親（→因子）")
    print("=" * 78)
    print("（lag≥1 のドライバ→因子＝向きが明確な先行リンク。lag0 は US→JP の時差で解釈）")
    factor_parents: dict[str, list[tuple[str, int, float]]] = {}
    for fj, fc in enumerate(factors):
        parents = []
        for i in range(len(allvars)):
            for lag in range(0, tau_max + 1):
                if lag == 0 and i == fj:
                    continue
                g = graph[i, fj, lag]
                if g in ("-->", "o->") and (lag >= 1 or name[i] in drivers):
                    parents.append((name[i], lag, float(val[i, fj, lag])))
        parents.sort(key=lambda t: -abs(t[2]))
        factor_parents[fc] = parents
        print(f"\n[{fc}] 親（|MCI| 降順）:")
        if not parents:
            print("   （有意な親なし）")
        for nm, lag, v in parents[:8]:
            kind = "ドライバ" if nm in drivers else "因子"
            print(f"   {nm:12} lag={lag}  MCI={v:+.3f}  ({kind})")

    # 素朴仕様との対比（docs/41 は factor~MKT のみ）
    print("\n→ docs/41 の素朴な仕様（因子~市場のみ）に対し、PCMCI+ は外生ドライバの"
          "ラグ付き親を同定＝調整集合が変わる（仕様の差が mirage を生む）。")

    # ---- Part 2: 因果効果の構造ブレイク vs パフォーマンスブレイク ---------
    print("\n" + "=" * 78)
    print("Part 2  因果効果(ドライバ→因子)の構造ブレイク vs パフォーマンスブレイク")
    print("=" * 78)
    W = 252
    print(f"各因子の最強ラグ付きドライバ親を選び、調整集合で因果効果をローリング推定"
          f"（窓{W}日）→ CUSUM。")

    alarms = {}
    for fc in factors:
        # ラグ≥1 のドライバ親のみ（PIT 明確）。無ければ lag0 ドライバを許容。
        dps = [(nm, lag, v) for (nm, lag, v) in factor_parents[fc]
               if nm in drivers and lag >= 1]
        if not dps:
            dps = [(nm, lag, v) for (nm, lag, v) in factor_parents[fc]
                   if nm in drivers]
        if not dps:
            print(f"\n[{fc}] ドライバ親なし → スキップ")
            continue
        target_nm, target_lag, target_v = dps[0]
        # 調整集合＝全ドライバ親（各自のラグでシフト）
        X = pd.DataFrame(index=panel.index)
        colnames = []
        for nm, lag, v in dps[:4]:
            col = f"{nm}_L{lag}"
            X[col] = panel[nm].shift(lag)
            colnames.append((col, nm, lag))
        target_col = f"{target_nm}_L{target_lag}"

        eff = rolling_causal_effect(panel[fc], X, target_col, W)
        perf = panel[fc].rolling(W).mean()
        a_eff = page_cusum(eff.diff().dropna(), k_sd=0.5, h_sd=5.0)
        a_perf = page_cusum(perf.diff().dropna(), k_sd=0.5, h_sd=5.0)
        alarms[fc] = {"eff": a_eff, "perf": a_perf, "driver": target_col}
        print(f"\n[{fc}]  主リンク {target_col}→{fc}  (MCI={target_v:+.3f})")
        print(f"  因果効果 構造変化 : {_fmt(a_eff)}")
        print(f"  パフォ   平均変化 : {_fmt(a_perf)}")

    # ---- Part 3: 先行性の検証（外生ドライバなら先行するか）---------------
    print("\n" + "=" * 78)
    print("Part 3  外生ドライバの因果効果ブレイクはパフォーマンスに先行するか？")
    print("=" * 78)
    print(f"{'factor':8}{'パフォ警報':>10}{'先行':>8}{'同時':>8}{'遅行':>8}{'孤立':>8}")
    LEAD = 120
    for fc in factors:
        if fc not in alarms:
            continue
        struct = sorted(pd.Timestamp(d) for d in alarms[fc]["eff"])
        lead = sim = lag = iso = 0
        for p in sorted(pd.Timestamp(d) for d in alarms[fc]["perf"]):
            near = [s for s in struct if abs((s - p).days) <= LEAD]
            if not near:
                iso += 1
                continue
            d = min(near, key=lambda s: abs((s - p).days))
            gap = (d - p).days
            if gap < -10:
                lead += 1
            elif gap > 10:
                lag += 1
            else:
                sim += 1
        print(f"{fc:8}{len(alarms[fc]['perf']):>10}{lead:>8}{sim:>8}{lag:>8}{iso:>8}")

    print("\n[参照] docs/27 事後“真の”転換点:", ", ".join(TRUE_BREAKS))
    print("解釈：docs/41 では価格由来の構造はパフォと同時（先行ゼロ）だった。"
          "外生ドライバの因果効果で『先行』が増えれば、監視対象として価値がある。")
    print("\n本診断は現象記述であり戦略でも判定でもない（K不変・先読みなし）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
