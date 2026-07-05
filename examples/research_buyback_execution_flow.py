"""research_buyback_execution_flow — 自社株買い実執行フロー（事前登録 docs/62）.

仮説: EDINET「自己株券買付状況報告書」を月 t に提出した企業（＝実執行中）は t+1 に forward 超過。
red-team 反映: submitDateTime アンカー・forward t+1（H-16）・対ユニバース＋サイズ(turnover)中立＋
プラセボ（H-18）・in-regime（2025-06+）＝機構実在判定（認定でなく）。

Stage-0（K=0・記述スキャン・registry 不参照）: §2.6.2 ゲート
  超過が (i) 正 (ii) プラセボ分布の外 (iii) turnover 中立でも正 → いずれか未達で KILL。
Stage-1（Stage-0 生存時のみ・judge_grid・K=3）: bef_exec_ls / bef_exec_sizematch / bef_persist_w2。
  ※事前登録の bef_placebo は Stage-0 の null、bef_exec_costed は judge_grid が全セルに 15bps を課すため
    別セル不要（K を消費しない・prereg §2 の設計冗長を実装で解消）。

実行: $env:PYTHONUTF8="1"; .venv/Scripts/python.exe examples/research_buyback_execution_flow.py
"""
from __future__ import annotations
import glob
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

SEED = 20260705
FWD = 1          # 提出月 t → t+1 リターン（H-16 forward）
NPLAC = 2000
MIN_EXEC = 10
DECILES = 10
SCOPE = "buyback_execution_flow"
START = "2025-07-01"     # 最初の保有月（2025-06 提出）
COST_BPS = 15.0


# ── EDINET executor firm-months（自己株券買付状況報告書）────────────────
def load_executors():
    pat = re.compile("自己株券買付状況")
    frames = []
    for x in sorted(glob.glob("data/edinet/list/*.parquet")):
        d = pd.read_parquet(x)
        if "docDescription" not in d.columns or len(d) == 0:
            continue
        m = d[d["docDescription"].astype(str).str.contains(pat, na=False)]
        if len(m):
            frames.append(m[["secCode", "submitDateTime"]])
    E = pd.concat(frames, ignore_index=True).dropna(subset=["secCode"])
    E["code"] = E["secCode"].astype(str).str.split(".").str[0].str.zfill(5)
    E["sm"] = pd.to_datetime(E["submitDateTime"]).dt.to_period("M")
    em = E[["sm", "code"]].drop_duplicates()
    exec_by_month = em.groupby("sm")["code"].apply(set).to_dict()
    persist = {t: exec_by_month[t] & exec_by_month.get(t - 1, set()) for t in exec_by_month}
    return E, exec_by_month, persist


def ir(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.nan if len(x) < 2 or x.std(ddof=1) == 0 else x.mean() / x.std(ddof=1) * np.sqrt(12)


def stage0(exec_by_month, persist):
    mpx = pd.read_parquet("data/processed/equities/wide/adj_close.parquet").resample("ME").last()
    ret = mpx.pct_change()
    ret.index = ret.index.to_period("M")
    price_pm = mpx.copy()
    price_pm.index = price_pm.index.to_period("M")
    mtov = pd.read_parquet("data/processed/equities/wide/turnover.parquet").resample("ME").mean()
    mtov.index = mtov.index.to_period("M")

    months = sorted(m for m in exec_by_month if (m + FWD) in ret.index and m in price_pm.index)
    rng = np.random.default_rng(SEED)
    rows, per_univ = [], []
    for t in months:
        r = ret.loc[t + FWD].dropna()
        pr_t = price_pm.loc[t]
        univ = [c for c in r.index if pd.notna(pr_t.get(c, np.nan))]
        r = r.loc[univ]
        ex = [c for c in exec_by_month[t] if c in r.index]
        if len(ex) < MIN_EXEC:
            continue
        uni_ret, ex_ret = r.mean(), r.loc[ex].mean()
        tv = mtov.loc[t].reindex(r.index) if t in mtov.index else pd.Series(np.nan, index=r.index)
        dn = np.nan
        if tv.notna().sum() > DECILES * 5:
            dec = pd.qcut(tv.rank(method="first"), DECILES, labels=False)
            exset = set(ex)
            num, den = 0.0, 0
            for dl in range(DECILES):
                idx = dec.index[dec == dl]
                e = [c for c in idx if c in exset]
                ne = [c for c in idx if c not in exset]
                if e and ne:
                    num += len(e) * (r.loc[e].mean() - r.loc[ne].mean())
                    den += len(e)
            dn = num / den if den else np.nan
        pex = [c for c in persist.get(t, set()) if c in r.index]
        psp = (r.loc[pex].mean() - uni_ret) if len(pex) >= MIN_EXEC else np.nan
        rows.append({"month": str(t), "n_exec": len(ex), "n_univ": len(r), "n_persist": len(pex),
                     "ex_ret": ex_ret, "uni_ret": uni_ret, "spread": ex_ret - uni_ret,
                     "dn_spread": dn, "persist_spread": psp})
        per_univ.append((r.values, len(ex)))
    res = pd.DataFrame(rows)

    plac = np.empty(NPLAC)
    for k in range(NPLAC):
        s = sum(rv[rng.integers(0, len(rv), ne)].mean() - rv.mean() for rv, ne in per_univ)
        plac[k] = s / len(per_univ)

    msp, mdn = res["spread"].mean(), res["dn_spread"].mean()
    p_val = float((plac >= msp).mean())
    print("=== buyback_execution_flow  Stage-0 (docs/62) ===")
    print(res.round(4).to_string(index=False))
    print(f"\n平均 生spread(exec-univ) : {msp:+.4%}/月  IR≈{ir(res['spread']):+.2f}  hit {100*(res['spread']>0).mean():.0f}%")
    print(f"平均 turnover中立 spread  : {mdn:+.4%}/月  IR≈{ir(res['dn_spread']):+.2f}  hit {100*(res['dn_spread']>0).mean():.0f}%")
    print(f"平均 persistence spread   : {res['persist_spread'].dropna().mean():+.4%}/月")
    print(f"プラセボ null mean {plac.mean():+.4%} sd {plac.std():.4%} p95 {np.percentile(plac,95):+.4%} → 片側p {p_val:.3f}")
    g = (msp > 0, p_val < 0.05, mdn > 0)
    print(f"\nゲート (i)生>0={'PASS' if g[0] else 'FAIL'} ({msp:+.4%})  "
          f"(ii)プラセボ外={'PASS' if g[1] else 'FAIL'} (p={p_val:.3f})  "
          f"(iii)turnover中立>0={'PASS' if g[2] else 'FAIL'} ({mdn:+.4%})")
    return all(g)


def stage1(exec_by_month, persist):
    from invest_system.data.store import load_wide
    from invest_system.equities.frictions import limit_lock_flags
    from invest_system.research import AsOfView, PrecomputedWeights, judge_grid, write_html
    from invest_system.validation.registry import default_registry
    try:
        from invest_system.data.sources import jquants as jq
        from invest_system.equities.universe import filter_common_stocks
        listed = jq.fetch_listed_info().assign(Code=lambda d: d["Code"].astype(str))
        common = set(filter_common_stocks(listed)["Code"])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] common-stock フィルタ不可（{e!r}）→ 全銘柄ユニバース")
        common = None

    adj, turn = load_wide("adj_close"), load_wide("turnover")
    close, high, low = load_wide("close"), load_wide("high"), load_wide("low")
    ul, ll, vol_ = load_wide("upper_limit"), load_wide("lower_limit"), load_wide("volume")
    cols = [c for c in adj.columns if (common is None or str(c) in common)]
    adj, turn, close, high, low, ul, ll, vol_ = (
        d.reindex(columns=cols) for d in (adj, turn, close, high, low, ul, ll, vol_))
    adj = adj.loc[START:]
    idx = adj.index
    dper = idx.to_period("M")
    traded = turn.reindex(idx).notna() & (turn.reindex(idx) > 0)
    liq_hist = traded.rolling(252, min_periods=60).sum() >= 60
    eligible = traded & liq_hist & adj.notna()
    tadv = turn.rolling(252, min_periods=60).mean()
    colpos = {c: i for i, c in enumerate(cols)}
    hold_months = sorted(set(dper))
    last_day = {M: idx[dper == M][-1] for M in hold_months}

    def build(long_by_month, kind):
        W = pd.DataFrame(0.0, index=idx, columns=cols)
        for M in hold_months:
            sig = M - 1
            L0 = long_by_month.get(sig)
            if not L0:
                continue
            dld = last_day.get(M)  # 保有月初の直前営業日＝決定情報は sig 月末までに公開
            elig = set(np.array(cols)[eligible.loc[dld].values]) if dld in eligible.index else set()
            L = [c for c in L0 if c in elig]
            if len(L) < MIN_EXEC:
                continue
            days = idx[dper == M]
            if kind == "univ":
                S = list(elig)
                W.loc[days, L] += 1.0 / len(L)
                W.loc[days, S] += -1.0 / len(S)
            elif kind == "sizematch":
                tv = tadv.loc[dld].reindex(cols)
                sub = tv[tv.index.isin(elig)].dropna()
                if len(sub) < DECILES * 5:
                    continue
                dec = pd.qcut(sub.rank(method="first"), DECILES, labels=False)
                Lset = set(L)
                wl = 1.0 / len(L)
                for c in L:
                    W.loc[days, c] += wl
                for dl in range(DECILES):
                    idxd = dec.index[dec == dl]
                    e = [c for c in idxd if c in Lset]
                    ne = [c for c in idxd if c not in Lset]
                    if e and ne:
                        W.loc[days, ne] += -(len(e) / len(L)) / len(ne)
        return W.replace(0.0, np.nan)

    strategies = [
        PrecomputedWeights(build(exec_by_month, "univ"), name="bef_exec_ls",
                           params={"anchor": "submit", "fwd": 1, "short": "univ_ew"}),
        PrecomputedWeights(build(exec_by_month, "sizematch"), name="bef_exec_sizematch",
                           params={"anchor": "submit", "fwd": 1, "short": "turnover_decile"}),
        PrecomputedWeights(build(persist, "univ"), name="bef_persist_w2",
                           params={"anchor": "submit", "fwd": 1, "persist": 2}),
    ]
    no_buy, no_sell = limit_lock_flags(close.loc[START:], high.loc[START:], low.loc[START:],
                                       ul.loc[START:], ll.loc[START:], vol_.loc[START:])
    view = AsOfView({"close": adj})
    hyp = ("EDINET『自己株券買付状況報告書』提出（＝実執行中）企業は t+1 に forward 超過。"
           "月次 executor バスケット・ロング×ユニバース/サイズマッチ・ショート（docs/62・"
           "方向は Clarke2022 FRL の執行フロー機構で事前固定・in-regime 2025-06+）")
    rat = ("実執行の大口・価格非感応な買いフローが t+1 の需給を支える（発表軸 shareholder_return❌"
           "とは別＝執行増分）。in-regime のみ＝認定でなく機構実在判定。docs/62 §2.6")
    with default_registry() as reg:
        v = judge_grid(strategies, view, scope=SCOPE, hypothesis=hyp, economic_rationale=rat,
                       registry=reg, costs_bps=COST_BPS, execution_lag=1, adv=turn.loc[START:],
                       participation=0.1, no_buy=no_buy, no_sell=no_sell)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{SCOPE}.html"))
    return v


def main():
    E, exec_by_month, persist = load_executors()
    print(f"executor filings: {len(E)} / distinct firms {E['code'].nunique()}")
    survive = stage0(exec_by_month, persist)
    print(f"\nStage-0 判定: {'SURVIVE→grid' if survive else 'KILL (K=0)'}")
    if survive:
        print("\n=== Stage-1 judge_grid（K=3）===")
        stage1(exec_by_month, persist)


if __name__ == "__main__":
    main()
