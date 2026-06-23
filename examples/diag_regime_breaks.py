"""レジーム転換の多角的検出（使い捨て診断・K不変・判定なし）。

data/ のローカルキャッシュのみを直接読む（J-Quants API キー不要・ネットワーク不要）。
「あらゆる側面・あらゆる手段」で日本株のレジーム転換時点を推定し、レポートする。

手段（全て PIT 不問の事後分析＝戦略検証ではなく現象の記述）：
  M1 市場ボラ・レジーム（TOPIX 実現ボラの三分位＋クラスタ）
  M2 トレンド/弱気相場（高値からのドローダウン・200日線）
  M3 構造変化検定：CUSUM（ドリフト変化）＋二分割セグメンテーション（平均/分散の変化点・BIC罰則）
  M4 マルコフ・レジームスイッチング（statsmodels・2状態・分散切替）→ 平滑確率・遷移・期待継続
  M5 因子リーダーシップ回転（mom/reversal/low-vol/skew 等の月次LS・12M ローリングSharpe）
  M6 クロスセクション分散・平均相関（リスクオン/オフ）
  X  プロジェクト自身の regime.parquet（vol_regime/trend_up）との突合

出力：各手段の転換時点 → 統合タイムライン（複数手段が一致する“合意点”）→ 頻度/継続の統計。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).resolve().parent.parent / "data"
FEAT = DATA / "features"


# ----------------------------------------------------------------------------
def _topix_monthly():
    df = pd.read_parquet(DATA / "jquants/indices/code_0000.parquet")
    s = df.dropna(subset=["Date"]).set_index("Date")["C"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _binseg_mean(x, max_bkps=8, min_size=6):
    """平均変化の二分割セグメンテーション（BIC 罰則）。返り値＝変化点の位置 index。"""
    x = np.asarray(x, float)
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def sse(a, b):                       # [a,b) の平均周りSSE
        if b <= a:
            return 0.0
        s = cs[b] - cs[a]
        s2 = cs2[b] - cs2[a]
        return float(s2 - s * s / (b - a))

    pen = np.log(n) * np.var(x)          # 1分割あたりの罰則（BIC 風）
    segs, bkps = [(0, n)], []
    for _ in range(max_bkps):
        best = None
        for (a, b) in segs:
            if b - a < 2 * min_size:
                continue
            base = sse(a, b)
            for c in range(a + min_size, b - min_size + 1):
                gain = base - sse(a, c) - sse(c, b)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, a, c, b)
        if best is None or best[0] < pen:
            break
        g, a, c, b = best
        bkps.append(c)
        segs = [s for s in segs if s != (a, b)] + [(a, c), (c, b)]
    return sorted(bkps)


def _cusum(x):
    """標準化 CUSUM。最大乖離の位置（=最も顕著なドリフト変化点）を返す。"""
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    if sd == 0:
        return None
    c = np.cumsum(x - x.mean()) / sd
    return int(np.argmax(np.abs(c))), float(np.max(np.abs(c)))


def _monthly_stock_returns():
    r = pd.read_parquet(FEAT / "returns.parquet")
    rm = (1.0 + r).resample("ME").prod() - 1.0     # 月次複利
    return rm


def _factor_ls(panel_name, rm, q=0.2, min_n=50):
    """因子パネル（日次wide・符号付き=高いほどロング）→ 月末ランク→翌月LSリターン。"""
    p = pd.read_parquet(FEAT / f"{panel_name}.parquet")
    pm = p.resample("ME").last()
    idx = pm.index.intersection(rm.index)
    pm, rmn = pm.reindex(idx), rm.reindex(idx)
    fwd = rmn.shift(-1)
    out = {}
    for t in idx[:-1]:
        row = pm.loc[t].dropna()
        fr = fwd.loc[t]
        row = row[row.index.intersection(fr.dropna().index)]
        if len(row) < min_n:
            continue
        k = max(1, int(len(row) * q))
        order = row.sort_values()
        lo = fr.reindex(order.index[:k]).mean()
        hi = fr.reindex(order.index[-k:]).mean()
        out[t] = float(hi - lo)
    return pd.Series(out).sort_index()


def _roll_sharpe(s, w=12, ann=12):
    return (s.rolling(w).mean() / s.rolling(w).std(ddof=1) * np.sqrt(ann))


def _fmt(dts):
    return ", ".join(pd.Timestamp(d).strftime("%Y-%m") for d in dts)


def main() -> int:
    print("=" * 78)
    print("レジーム転換 多角的診断（使い捨て・K不変・data/ キャッシュのみ）")
    print("=" * 78)

    tx = _topix_monthly()
    txm = tx.resample("ME").last()
    ret = txm.pct_change().dropna()
    print(f"\nTOPIX 月次: {ret.index.min():%Y-%m} 〜 {ret.index.max():%Y-%m}  "
          f"({len(ret)} か月)")

    consensus = {}   # date(month) -> [methods]

    def flag(ts, tag):
        m = pd.Timestamp(ts).to_period("M")
        consensus.setdefault(m, []).append(tag)

    # --- M1 ボラ・レジーム ---------------------------------------------------
    vol = ret.rolling(6).std() * np.sqrt(12)
    hi_thr, lo_thr = vol.quantile(0.66), vol.quantile(0.33)
    vstate = pd.Series(np.where(vol > hi_thr, "HIGH",
                       np.where(vol < lo_thr, "LOW", "MID")), index=vol.index)
    sw = vstate[vstate != vstate.shift(1)].dropna()
    print("\n[M1] 市場ボラ・レジーム（6M実現ボラ年率の三分位）")
    print(f"  高ボラ閾値≈{hi_thr:.1%} / 低ボラ閾値≈{lo_thr:.1%}")
    print("  状態遷移:", "  ".join(f"{d:%Y-%m}->{s}" for d, s in sw.items()))
    for d, s in sw.items():
        if s == "HIGH":
            flag(d, "vol↑")

    # --- M2 トレンド/弱気相場（ドローダウン・エピソード統合）------------------
    cum = (1 + ret).cumprod()
    dd = cum / cum.cummax() - 1.0
    episodes, in_ep, start, tr, tv = [], False, None, None, 0.0
    for d, v in dd.items():
        if not in_ep:
            if v < -0.10:                    # -10% 突破でエピソード開始
                in_ep, start, tr, tv = True, d, d, v
        else:
            if v < tv:
                tr, tv = d, v
            if v > -0.03:                    # -3% まで回復で終了
                if tv <= -0.15:              # 材料的な弱気のみ採用
                    episodes.append((start, tr, tv))
                in_ep = False
    if in_ep and tv <= -0.15:
        episodes.append((start, tr, tv))
    print("\n[M2] 弱気相場（高値から-15%超のドローダウン・エピソード統合）")
    for st, t, mn in episodes:
        print(f"  下落入り {st:%Y-%m} → 底 {t:%Y-%m} ({mn:.0%})")
        flag(t, "DD↓")

    # --- M3 構造変化検定 -----------------------------------------------------
    bk = _binseg_mean(ret.values)
    bk_mean = [ret.index[i] for i in bk]
    bk_v = _binseg_mean((ret - ret.mean()).abs().values)
    bk_vol = [ret.index[i] for i in bk_v]
    cu = _cusum(ret.values)
    print("\n[M3] 構造変化検定")
    print(f"  二分割(平均リターン)変化点: {_fmt(bk_mean) or '—'}")
    print(f"  二分割(ボラ |ret|)変化点 : {_fmt(bk_vol) or '—'}")
    if cu:
        print(f"  CUSUM 最大ドリフト変化点 : {ret.index[cu[0]]:%Y-%m} (|C|={cu[1]:.1f})")
        flag(ret.index[cu[0]], "CUSUM")
    for d in bk_mean:
        flag(d, "break(μ)")
    for d in bk_vol:
        flag(d, "break(σ)")

    # --- M4 マルコフ・レジームスイッチング -----------------------------------
    print("\n[M4] マルコフ・レジームスイッチング（2状態・分散切替）")
    try:
        from statsmodels.tsa.regime_switching.markov_regression import (
            MarkovRegression,
        )
        y = (ret * 100).astype(float)
        mod = MarkovRegression(y.values, k_regimes=2, trend="c",
                               switching_variance=True)
        res = mod.fit(em_iter=100, search_reps=30, disp=False)
        probs = np.asarray(res.smoothed_marginal_probabilities)
        if probs.shape[0] != len(y):                       # 向きガード
            probs = probs.T
        rv = ret.values                                    # 高分散レジーム=波乱
        var_by = []
        for k in range(2):
            w = probs[:, k]
            mu = np.average(rv, weights=w)
            var_by.append(np.average((rv - mu) ** 2, weights=w))
        turb = int(np.argmax(var_by))
        sp = pd.Series(probs[:, turb], index=y.index)
        state = (sp > 0.5).astype(int)
        trans = state[state != state.shift(1)].dropna()
        try:
            durs = res.expected_durations
            print(f"  波乱状態の期待継続≈{durs[turb]:.1f}か月 / "
                  f"平穏≈{durs[1 - turb]:.1f}か月")
        except Exception:                                  # noqa: BLE001
            pass
        print("  波乱状態(平滑確率>0.5)への突入/離脱:")
        for d, v in trans.items():
            kind = "→波乱" if v == 1 else "→平穏"
            print(f"    {pd.Timestamp(d):%Y-%m} {kind}")
            if v == 1:
                flag(d, "Markov波乱")
    except Exception as e:                                  # noqa: BLE001
        print("  （Markov 収束せず・スキップ）", e)

    # --- M5 因子リーダーシップ回転 -------------------------------------------
    print("\n[M5] 因子リーダーシップ回転（月次LS・12Mローリング年率Sharpe）")
    rm = _monthly_stock_returns()
    factors = {"momentum": "momentum_12_1", "reversal_1m": "mom_1m_reversal",
               "low_vol": "ivol", "low_beta": "beta", "skew": "ret_skew",
               "max_ret": "max_ret"}
    ls = {}
    for nm, panel in factors.items():
        try:
            ls[nm] = _factor_ls(panel, rm)
        except Exception as e:                              # noqa: BLE001
            print(f"  （{nm} 構築失敗: {e}）")
    lsdf = pd.DataFrame(ls).dropna(how="all")
    rs = lsdf.apply(_roll_sharpe).dropna(how="all")
    lead = rs.idxmax(axis=1).dropna()
    lead_sw = lead[lead != lead.shift(1)].dropna()
    print("  各因子の全期間 年率Sharpe:",
          "  ".join(f"{k}={v.mean()/v.std(ddof=1)*np.sqrt(12):+.2f}"
                    for k, v in lsdf.items() if v.std(ddof=1) > 0))
    print("  リーダー因子の交代（12M Sharpe 最上位の変化）:")
    for d, f in lead_sw.items():
        print(f"    {pd.Timestamp(d):%Y-%m} → {f}")
        flag(d, "因子交代")

    # --- M6 分散・平均相関 ---------------------------------------------------
    print("\n[M6] クロスセクション分散・平均相関（リスクオン/オフ）")
    r = pd.read_parquet(FEAT / "returns.parquet")
    disp = r.std(axis=1).resample("ME").mean()
    bkd = _binseg_mean(disp.dropna().values)
    bkd_d = [disp.dropna().index[i] for i in bkd]
    print(f"  日次ディスパージョン(銘柄間std)の月次平均: 変化点 {_fmt(bkd_d) or '—'}")
    for d in bkd_d:
        flag(d, "分散変化")

    # --- X プロジェクト regime.parquet 突合 ----------------------------------
    print("\n[X] プロジェクト regime.parquet（vol_regime / trend_up）の遷移")
    try:
        rg = pd.read_parquet(FEAT / "regime.parquet")
        for col in ("vol_regime", "trend_up"):
            s = rg[col].resample("ME").last().dropna()
            t = s[s != s.shift(1)].dropna()
            print(f"  {col}: {len(t)} 回遷移  最近=",
                  "  ".join(f"{d:%Y-%m}:{int(v)}" for d, v in t.tail(6).items()))
    except Exception as e:                                  # noqa: BLE001
        print("  （regime.parquet 読めず）", e)

    # --- 統合タイムライン（±2か月で群化・手段ファミリで合意度を採点）----------
    fam = {"vol↑": "ボラ", "break(σ)": "ボラ", "分散変化": "ボラ",
           "DD↓": "下落", "CUSUM": "トレンド", "break(μ)": "トレンド",
           "Markov波乱": "マルコフ", "因子交代": "因子"}
    items = sorted(consensus.items())                     # [(Period('M'), [tags])]
    clusters = []
    for m, tags in items:
        placed = False
        for cl in clusters:
            if abs(m.ordinal - cl["center"].ordinal) <= 2:
                cl["months"].append(m)
                cl["tags"].update(tags)
                cl["fams"].update(fam.get(t, t) for t in tags)
                placed = True
                break
        if not placed:
            clusters.append({"center": m, "months": [m], "tags": set(tags),
                             "fams": {fam.get(t, t) for t in tags}})
    print("\n" + "=" * 78)
    print("統合タイムライン（±2か月で群化・独立した“手段ファミリ”数で確度を採点）")
    print("=" * 78)
    for cl in clusters:
        lo, hi = min(cl["months"]), max(cl["months"])
        span = f"{lo}" if lo == hi else f"{lo}〜{hi}"
        nf = len(cl["fams"])
        mark = "★★" if nf >= 4 else ("★" if nf == 3 else ("·" if nf == 2 else "  "))
        print(f"  {mark:<2} {span:<17} [{nf}ファミリ] "
              f"{', '.join(sorted(cl['fams']))}")
    strong = [cl for cl in clusters if len(cl["fams"]) >= 3]
    print("\n  ★主要レジーム転換（3ファミリ以上が一致）:")
    for cl in strong:
        lo, hi = min(cl["months"]), max(cl["months"])
        span = f"{lo}" if lo == hi else f"{lo}〜{hi}"
        print(f"    - {span}  ({len(cl['fams'])}ファミリ: "
              f"{', '.join(sorted(cl['fams']))})")

    # --- 頻度・継続 ----------------------------------------------------------
    nsw = int((vstate != vstate.shift(1)).sum())
    print(f"\n[頻度] 市場ボラ・レジーム遷移={nsw}回／{len(ret)}か月"
          f"（≈{len(ret)/max(1,nsw):.1f}か月に1回）。"
          f"主要転換={len(strong)}回／{len(ret)/12:.0f}年"
          f"（≈{len(ret)/12/max(1,len(strong)):.1f}年に1回）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
