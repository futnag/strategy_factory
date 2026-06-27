"""日足の因果的レジーム検知・仕様感応性の診断（使い捨て・K不変・判定なし）。

López de Prado の causal factor investing（specification error / collider bias /
factor mirage）と「パフォーマンスのブレイクを待つのではなく因果関係の structural
break を監視せよ」という考え方を、**日足の価格系ファクター LS リターン**で実験する。
すべて data/ のローカルキャッシュのみ（API キー不要・オフライン・先読みなし）。

> 制約：オフライン・キャッシュに book/market 値パネルが無いため value/PEAD の LS は
> 再構築できない（docs/27 §8）。本診断は docs/27 M5 と同系の**価格系ファクター**
> （モメンタム/短期リバーサル/ロー・ボラ）＋市場で因果構造を扱う。

二部構成：
  Part A  仕様感応性 / collider bias（= factor mirage の機構）
     A1 周辺相関 vs 偏相関（市場・市場ボラで条件付け）で関係が変わることを示す
     A2 collider（共通結果）への条件付けが**見せかけの相関**を生むことを
        (i) 合成データ（決定的・機構証明）と (ii) 実データ（市場=共通結果）で示す
  Part B  因果構造ブレイク vs パフォーマンスブレイク（監視対象の比較）
     B1 ローリング窓で各ファクターの「市場への因果係数 βmkt」と
        「他ファクターとの偏相関」を推定し、その**構造変化**を CUSUM で検知
     B2 同じ窓で**パフォーマンス**（平均 LS リターン）の変化を CUSUM で検知
     B3 因果ブレイク警報がパフォーマンス警報に**先行/同時/遅行**かを比較し、
        docs/27 の事後“真の”転換点と突き合わせる

判定（DSR/PASS-FAIL）は一切しない。これは現象記述の診断であり戦略ではない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).resolve().parent.parent / "data"
FEAT = DATA / "features"

# docs/27 で確定した“真の”構造転換（事後・複数手段の合意）— 参照用
TRUE_BREAKS = ["2018-10", "2020-03", "2020-12", "2022-09"]


# ----------------------------------------------------------- ファクター LS 構築
def _winsorize(df: pd.DataFrame, lo=0.01, hi=0.99) -> pd.DataFrame:
    """行ごと（日ごと）に分位クリップ。外れ値（returns に最大 600% 等）対策。"""
    q = df.quantile([lo, hi], axis=1).T
    return df.clip(lower=q[lo], upper=q[hi], axis=0)


def ls_return(signal: pd.DataFrame, fwd_ret: pd.DataFrame,
              quantile: float = 0.2, sign: float = 1.0) -> pd.Series:
    """日次 LS リターン（PIT）。t 時点の signal で上位/下位 quantile を等加重 L/S し、
    **翌日** fwd_ret を実現。sign=-1 で向きを反転（ロー・ボラ＝低值ロング等）。

    strategy.CrossSectionalFactor と同じ上位ロング/下位ショート規約（quantile=0.2）。
    返り値は実現日（t+1）に index 付け。
    """
    sig = (signal * sign)
    common = sig.columns.intersection(fwd_ret.columns)
    sig, fr = sig[common], fwd_ret[common]
    out = {}
    for t, row in sig.iterrows():
        row = row.dropna()
        if len(row) < 20:
            continue
        k = max(1, int(len(row) * quantile))
        order = row.sort_values()
        longs, shorts = order.index[-k:], order.index[:k]
        out[t] = (longs, shorts)
    # 翌日リターンへ写像
    days = fr.index
    pos = {d: i for i, d in enumerate(days)}
    res = {}
    for t, (longs, shorts) in out.items():
        i = pos.get(t)
        if i is None or i + 1 >= len(days):
            continue
        nxt = days[i + 1]
        rl = fr.loc[nxt, longs].mean()
        rs = fr.loc[nxt, shorts].mean()
        if np.isfinite(rl) and np.isfinite(rs):
            res[nxt] = float(rl - rs)
    return pd.Series(res).sort_index()


# --------------------------------------------------------------- 偏相関ユーティリティ
def _resid(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """y を Z（定数項付き）に回帰した残差。Z が空なら中心化のみ。"""
    if Z.size == 0:
        return y - y.mean()
    A = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def pcorr(x: pd.Series, y: pd.Series, Z: pd.DataFrame | None = None) -> float:
    """偏相関 corr(x, y | Z)。Z=None で周辺相関。共通 index で揃える。"""
    cols = [x, y] + ([Z[c] for c in Z.columns] if Z is not None else [])
    df = pd.concat(cols, axis=1).dropna()
    if len(df) < 30:
        return np.nan
    xv = df.iloc[:, 0].values
    yv = df.iloc[:, 1].values
    Zv = df.iloc[:, 2:].values if (Z is not None and Z.shape[1] > 0) else np.empty((len(df), 0))
    rx, ry = _resid(xv, Zv), _resid(yv, Zv)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.mean(rx * ry) / (sx * sy))


# ------------------------------------------------------------------ CUSUM（≤t）
def page_cusum(x: pd.Series, k_sd=0.5, h_sd=4.0, min_periods=60):
    """両側 Page CUSUM。基準平均/分散は ≤t 拡張窓（因果）。警報位置を返しリセット。"""
    mu = x.expanding(min_periods=min_periods).mean().shift(1)
    sd = x.expanding(min_periods=min_periods).std().shift(1)
    s_hi = s_lo = 0.0
    alarms = []
    for t, v in x.items():
        m, d = mu.get(t), sd.get(t)
        if not (np.isfinite(m) and np.isfinite(d) and np.isfinite(v) and d > 0):
            continue
        z = (v - m) / d
        s_hi = max(0.0, s_hi + z - k_sd)
        s_lo = min(0.0, s_lo + z + k_sd)
        if s_hi > h_sd or s_lo < -h_sd:
            alarms.append(t)
            s_hi = s_lo = 0.0
    return alarms


def _fmt_dates(ds, n=8):
    ds = list(ds)
    s = "  ".join(f"{pd.Timestamp(d):%Y-%m}" for d in ds[:n])
    return (s + (f"  …(+{len(ds)-n})" if len(ds) > n else "")) or "—"


# ===========================================================================
def main() -> int:
    print("=" * 78)
    print("日足の因果的レジーム検知・仕様感応性の診断（K不変・判定なし・オフライン）")
    print("=" * 78)

    ret = pd.read_parquet(FEAT / "returns.parquet")
    ret = _winsorize(ret)
    mom = pd.read_parquet(FEAT / "momentum_12_1.parquet")
    rev = pd.read_parquet(FEAT / "reversal_5.parquet")
    vol = pd.read_parquet(FEAT / "vol_20.parquet")
    print(f"日次 {ret.index.min():%Y-%m-%d}〜{ret.index.max():%Y-%m-%d}  "
          f"{len(ret)}営業日 / 銘柄~{int(ret.notna().sum(axis=1).median())}")

    # --- 日次ファクター LS リターン（PIT・価格系・docs/27 M5 同系）-------------
    f = pd.DataFrame({
        "MOM":    ls_return(mom, ret, sign=+1.0),   # 高モメンタム ロング
        "REV":    ls_return(rev, ret, sign=-1.0),   # 短期リバーサル（低リターン ロング）
        "LOWVOL": ls_return(vol, ret, sign=-1.0),   # 低ボラ ロング
    }).dropna(how="all")
    f["MKT"] = ret.mean(axis=1).reindex(f.index)     # 等加重市場リターン（共通結果候補）
    f["MKTVOL"] = (f["MKT"].rolling(20).std() * np.sqrt(252))  # 市場ボラ（条件変数）
    f = f.dropna()
    print(f"ファクター LS 期間 {f.index.min():%Y-%m-%d}〜{f.index.max():%Y-%m-%d}  "
          f"{len(f)}日")
    ann = lambda s: (s.mean() * 252, s.std() * np.sqrt(252),
                     s.mean() / s.std() * np.sqrt(252))
    print("\nファクター年率 (mean / vol / Sharpe):")
    for c in ["MOM", "REV", "LOWVOL", "MKT"]:
        m, v, sh = ann(f[c])
        print(f"  {c:7} {m:+7.1%} / {v:6.1%} / SR {sh:+.2f}")

    # ====================================================================
    # Part A — 仕様感応性 / collider bias（factor mirage の機構）
    # ====================================================================
    print("\n" + "=" * 78)
    print("Part A  仕様感応性 / collider bias（factor mirage の機構）")
    print("=" * 78)

    facs = ["MOM", "REV", "LOWVOL"]
    print("\n[A1] ファクター間の相関：条件付けで関係が変わる（仕様感応性）")
    print(f"{'pair':14}{'周辺':>9}{'|MKT':>9}{'|MKT,VOL':>11}")
    for i in range(len(facs)):
        for j in range(i + 1, len(facs)):
            a, b = facs[i], facs[j]
            r0 = pcorr(f[a], f[b])
            r1 = pcorr(f[a], f[b], f[["MKT"]])
            r2 = pcorr(f[a], f[b], f[["MKT", "MKTVOL"]])
            print(f"{a+'~'+b:14}{r0:+9.3f}{r1:+9.3f}{r2:+11.3f}")
    print("  → 条件集合で符号・強度が動く＝『関係が有るか』は仕様依存（mirage の温床）。")

    print("\n[A2-i] collider 機構の証明（合成・決定的 seed）：")
    rng = np.random.default_rng(0)
    n = 4000
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)              # u ⟂ v（真に独立）
    collider = u + v + 0.3 * rng.standard_normal(n)   # 共通結果（collider）
    U, V, C = (pd.Series(u), pd.Series(v), pd.DataFrame({"C": collider}))
    print(f"  corr(u,v) 周辺      = {pcorr(U, V):+.3f}  （真の独立 ≈ 0）")
    print(f"  corr(u,v | collider)= {pcorr(U, V, C):+.3f}  "
          f"（共通結果で条件付け → 見せかけの負相関が発生）")
    print("  → collider への『過剰な条件付け』が偽の関係を生む（LdP の警告そのもの）。")

    print("\n[A2-ii] 実データの collider：市場リターンは多くのファクターの共通結果。")
    for i in range(len(facs)):
        for j in range(i + 1, len(facs)):
            a, b = facs[i], facs[j]
            r0, r1 = pcorr(f[a], f[b]), pcorr(f[a], f[b], f[["MKT"]])
            flag = "  ★条件付けで|相関|増" if abs(r1) > abs(r0) + 0.03 else ""
            print(f"  {a+'~'+b:14} 周辺 {r0:+.3f} → |MKT {r1:+.3f}{flag}")
    print("  → 市場（共通結果）で機械的に条件付けると関係が歪む。"
          "条件集合は因果図から選ぶべき。")

    # ====================================================================
    # Part B — 因果構造ブレイク vs パフォーマンスブレイク
    # ====================================================================
    print("\n" + "=" * 78)
    print("Part B  因果構造ブレイク vs パフォーマンスブレイク（監視対象の比較）")
    print("=" * 78)
    W = 252
    print(f"ローリング窓 {W}営業日。各ファクターで構造系列とパフォ系列を作り CUSUM。")

    # B 用：構造系列＝βmkt（factor~MKT のローリング係数）と、他ファクターとの
    # ローリング偏相関（|MKT,VOL 条件付け）。パフォ系列＝ローリング平均 LS リターン。
    def rolling_beta(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
        cov = y.rolling(w).cov(x)
        var = x.rolling(w).var()
        return (cov / var).rename("beta")

    def rolling_pcorr(a: pd.Series, b: pd.Series, Z: pd.DataFrame, w: int) -> pd.Series:
        idx = a.index
        out = pd.Series(np.nan, index=idx)
        arr_a, arr_b = a.values, b.values
        Zv = Z.values
        for e in range(w, len(idx) + 1):
            s = e - w
            sub = slice(s, e)
            ra = _resid(arr_a[sub], Zv[sub])
            rb = _resid(arr_b[sub], Zv[sub])
            sa, sb = ra.std(), rb.std()
            if sa > 0 and sb > 0:
                out.iloc[e - 1] = float(np.mean(ra * rb) / (sa * sb))
        return out

    other = {"MOM": "REV", "REV": "MOM", "LOWVOL": "MOM"}
    alarms = {}
    for c in facs:
        beta = rolling_beta(f[c], f["MKT"], W)
        pc = rolling_pcorr(f[c], f[other[c]], f[["MKT", "MKTVOL"]], W)
        perf = f[c].rolling(W).mean()
        # 構造ブレイク＝βmkt と 偏相関 の CUSUM（差分系列に対して）
        a_beta = page_cusum(beta.diff().dropna(), k_sd=0.5, h_sd=5.0)
        a_pc = page_cusum(pc.diff().dropna(), k_sd=0.5, h_sd=5.0)
        a_perf = page_cusum(perf.diff().dropna(), k_sd=0.5, h_sd=5.0)
        alarms[c] = {"βmkt": a_beta, "pcorr": a_pc, "perf": a_perf}
        print(f"\n[{c}]")
        print(f"  構造  βmkt 変化  : {_fmt_dates(a_beta)}")
        print(f"  構造  偏相関変化 : {_fmt_dates(a_pc)}")
        print(f"  パフォ 平均変化  : {_fmt_dates(a_perf)}")

    # B3 先行/同時/遅行：各パフォ警報の直近±に構造警報があるか（120営業日窓）
    print("\n[B3] 構造警報はパフォ警報に先行するか？（各パフォ警報の前後120営業日）")
    print(f"{'factor':8}{'パフォ警報数':>10}{'先行':>8}{'同時':>8}{'遅行':>8}{'孤立':>8}")
    LEAD = 120
    for c in facs:
        struct = sorted(pd.Timestamp(d) for d in (alarms[c]["βmkt"] + alarms[c]["pcorr"]))
        lead = sim = lag = iso = 0
        for p in sorted(pd.Timestamp(d) for d in alarms[c]["perf"]):
            near = [s for s in struct if abs((s - p).days) <= LEAD]
            if not near:
                iso += 1
            else:
                d = min(near, key=lambda s: abs((s - p).days))
                gap = (d - p).days
                if gap < -10:
                    lead += 1
                elif gap > 10:
                    lag += 1
                else:
                    sim += 1
        np_ = len(alarms[c]["perf"])
        print(f"{c:8}{np_:>10}{lead:>8}{sim:>8}{lag:>8}{iso:>8}")

    print("\n[参照] docs/27 事後“真の”転換点:", ", ".join(TRUE_BREAKS))
    print("注：構造系列(βmkt/偏相関)も価格から派生するため、暴落型転換では"
          "パフォと同時に動きやすい。『先行』が多いほど監視対象として価値がある。")
    print("\n本診断は現象記述であり戦略でも判定でもない（K不変・先読みなし）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
