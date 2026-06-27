"""因果調整によるファクター・プレミアムの不偏推定と安定性検証（K不変・判定なし）。

docs/42 §4-1/§5 の昇格パス。LdP の主張の本筋＝「レジームを当てる」ではなく
「**仕様を正す**（collider/confounder を因果図で判別し、正しい調整集合でファクター・
プレミアムを不偏推定する）」を、phase4 の月次横断面設計行列で一度きり検証する。

問い：value（book_to_market / earnings_yield）と PEAD（sue）のプレミアムは、
  - **素朴**な単変量推定、
  - **全部入り**（60特性で統制＝collider/mediator まで巻き込む過剰統制）、
  - **因果調整集合**（PC で学習した因果図から backdoor で選んだ統制集合）、
の間でどれだけ動くか（＝仕様感応性＝mirage）。そして docs/27 の4エポックで
**符号・大きさが最も安定するのはどの仕様か**（LdP は因果調整が最も安定と予測）。

すべて data/ のローカルキャッシュのみ（API キー不要・オフライン・先読みなし）。
判定（DSR/PASS-FAIL）はしない。これは戦略性能の主張ではなく**推定方法**の診断。

> docs/27/28・41/42 の「value/PEAD LS はオフライン再構築不可（B/M パネル無し）」は
> **日次 features キャッシュ**に限った話。月次 phase4 GKX 行列には book_to_market・
> earnings_yield・sue が揃っており、本診断はそれを使う。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "phase4"

# docs/27 の4エポック（事後・構造転換で区切る）
EPOCHS = [
    ("A 平穏ブル",        "2016-07-01", "2018-09-30"),
    ("B 後期→COVID",      "2018-10-01", "2020-03-31"),
    ("C コロナ後リフレ",  "2020-04-01", "2022-08-31"),
    ("D 利上げ/value復権", "2022-09-01", "2026-06-30"),
]

# 因果探索に使う代表ファクター（各ファミリ1代表＋成果 y）。
# outcome を覗いた選択ではなく**ファクター分類**で選ぶ（in-sample 選択ではない）。
DISCOVERY_VARS = [
    "book_to_market", "earnings_yield",      # value
    "sue_recent", "forecast_revision",       # PEAD
    "momentum", "mom_1m_reversal",           # trend/reversal
    "size", "amihud_illiq",                  # size/liquidity
    "roe", "gross_profitability", "accruals",  # quality
    "beta", "ivol",                          # risk
    "asset_growth", "sales_growth",          # growth/investment
]
TARGETS = ["book_to_market", "earnings_yield", "sue_recent"]


# --------------------------------------------------------- 横断面標準化（PIT）
def cs_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """各月（date）内でランク→z 標準化（ベクトル化）。横断面なので先読みなし。
    ランク化で外れ値は構造的に除去されるため別途の winsorize は不要。"""
    r = df.groupby(level="date").rank()
    mu = r.groupby(level="date").transform("mean")
    sd = r.groupby(level="date").transform("std").replace(0.0, np.nan)
    return (r - mu) / sd


# ----------------------------------------------------- Fama-MacBeth プレミアム
def fama_macbeth(y: pd.Series, X: pd.DataFrame, target: str):
    """月次横断面 OLS の傾きを平均（FM）。target 係数の平均・NW t・月次系列を返す。"""
    cols = list(X.columns)
    ti = cols.index(target)
    slopes = {}
    for dt, idx in X.groupby(level="date").groups.items():
        xs = X.loc[idx].values
        ys = y.loc[idx].values
        m = np.isfinite(ys) & np.isfinite(xs).all(axis=1)
        if m.sum() < len(cols) + 10:
            continue
        A = np.column_stack([np.ones(m.sum()), xs[m]])
        try:
            b, *_ = np.linalg.lstsq(A, ys[m], rcond=None)
        except np.linalg.LinAlgError:
            continue
        slopes[dt] = b[1 + ti]
    s = pd.Series(slopes).sort_index()
    if len(s) < 6:
        return np.nan, np.nan, s
    mean = s.mean()
    # Newey-West（lag=6）t 値
    x = (s - mean).values
    n = len(x)
    g0 = np.mean(x * x)
    var = g0
    for L in range(1, 7):
        w = 1.0 - L / 7.0
        cov = np.mean(x[L:] * x[:-L])
        var += 2 * w * cov
    se = np.sqrt(var / n)
    t = mean / se if se > 0 else np.nan
    return mean, t, s


def _epoch_slice(s: pd.Series, a, b) -> pd.Series:
    return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]


# ===========================================================================
def main() -> int:
    warnings.filterwarnings("ignore")
    print("=" * 78)
    print("因果調整によるファクター・プレミアム不偏推定と安定性（K不変・判定なし）")
    print("=" * 78)

    X = pd.read_parquet(DATA / "X.parquet")
    y = pd.read_parquet(DATA / "y.parquet")["y"]
    print(f"phase4 横断面：{X.shape[0]:,}行 × {X.shape[1]}特性 / "
          f"{X.index.get_level_values('date').nunique()}か月 "
          f"({X.index.get_level_values('date').min():%Y-%m}〜"
          f"{X.index.get_level_values('date').max():%Y-%m})")

    Xz = cs_standardize(X)
    yz = y.copy()

    # ---- Part 1: 因果図学習（PC・causal-learn）----------------------------
    print("\n" + "=" * 78)
    print("Part 1  因果図の学習（PC・Fisher-Z）→ 各 target の調整集合")
    print("=" * 78)
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    dv = [v for v in DISCOVERY_VARS if v in Xz.columns]
    disc = pd.concat([Xz[dv], yz.rename("y")], axis=1).dropna()
    # 計算量を抑えるためサブサンプル（行は CI 検定の検出力のみに効く）
    if len(disc) > 40000:
        disc = disc.sample(40000, random_state=0).sort_index()
    print(f"探索変数 {len(dv)}＋y / 標本 {len(disc):,}行・alpha=0.01 …")
    cg = pc(disc.values, alpha=0.01, indep_test=fisherz, show_progress=False)
    names = list(dv) + ["y"]
    yi = len(names) - 1
    adj = cg.G.graph  # CPDAG 隣接（-1/1 で有向、両端1で無向）

    def neighbors(i):
        out = []
        for j in range(len(names)):
            if j == i:
                continue
            if adj[i, j] != 0 or adj[j, i] != 0:
                out.append(j)
        return out

    y_nb = set(neighbors(yi))
    adjustment = {}
    for tg in TARGETS:
        if tg not in names:
            continue
        ti = names.index(tg)
        # backdoor 近似：target と y の**両方**に隣接する変数＝交絡候補。
        # target のみに隣接する変数（target の子＝collider/mediator 候補）は除く。
        conf = [names[j] for j in neighbors(ti) if j in y_nb and names[j] != "y"]
        adjustment[tg] = conf
        ty = "直接" if ti in y_nb else "間接/無"
        print(f"\n[{tg}] y との隣接: {ty}")
        print(f"   調整集合（target と y の共通隣接＝交絡候補）: "
              f"{conf if conf else '（なし）'}")

    # ---- Part 2: 3仕様でのプレミアム（全期間）----------------------------
    print("\n" + "=" * 78)
    print("Part 2  プレミアム推定：素朴 / 因果調整 / 全部入り（全期間 FM・NW-t）")
    print("=" * 78)
    allcols = list(Xz.columns)
    specs_premium = {}
    for tg in TARGETS:
        # 素朴（単変量）
        m_n, t_n, s_n = fama_macbeth(yz, Xz[[tg]], tg)
        # 因果調整集合
        adj_cols = [tg] + [c for c in adjustment.get(tg, []) if c in allcols]
        m_c, t_c, s_c = fama_macbeth(yz, Xz[adj_cols], tg)
        # 全部入り（60特性・過剰統制）
        m_k, t_k, s_k = fama_macbeth(yz, Xz[allcols], tg)
        specs_premium[tg] = {"素朴": s_n, "因果": s_c, "全部": s_k}
        print(f"\n[{tg}]  （月次プレミアム平均×100・NW-t）")
        print(f"   素朴(単変量)        : {m_n*100:+.3f}  t={t_n:+.2f}")
        print(f"   因果調整({len(adj_cols)-1}統制) : {m_c*100:+.3f}  t={t_c:+.2f}")
        print(f"   全部入り(59統制)    : {m_k*100:+.3f}  t={t_k:+.2f}")
        spread = max(abs(m_n), abs(m_c), abs(m_k)) - min(abs(m_n), abs(m_c), abs(m_k))
        print(f"   → 仕様間の|大きさ|レンジ {spread*100:.3f}（mirage の幅）")

    # ---- Part 3: エポック別の符号・大きさ安定性 --------------------------
    print("\n" + "=" * 78)
    print("Part 3  4エポック別プレミアム（×100）と安定性：仕様間比較")
    print("=" * 78)
    print("LdP 予測：因果調整が最も符号一貫・低変動。素朴/全部入りは局面で反転しやすい。")
    for tg in TARGETS:
        print(f"\n[{tg}]")
        print(f"  {'spec':12}" + "".join(f"{e[0][:8]:>10}" for e in EPOCHS)
              + f"{'符号一致':>8}{'CV':>7}")
        for spec_name, s in specs_premium[tg].items():
            cells, signs, vals = "", [], []
            for _, a, b in EPOCHS:
                sub = _epoch_slice(s, a, b)
                if len(sub) >= 3:
                    v = sub.mean() * 100
                    cells += f"{v:>+10.3f}"
                    signs.append(np.sign(v))
                    vals.append(v)
                else:
                    cells += f"{'—':>10}"
            consist = (f"{int(abs(sum(signs)))}/{len(signs)}"
                       if signs else "—")
            cv = (np.std(vals) / abs(np.mean(vals))
                  if vals and abs(np.mean(vals)) > 1e-9 else np.nan)
            print(f"  {spec_name:12}{cells}{consist:>8}{cv:>7.2f}")
    print("\n  符号一致 = 4エポックで同符号の数（4/4＝全期間で符号不変＝安定）。")
    print("  CV = |変動係数|（小さいほど大きさが安定）。")

    print("\n本診断は推定方法の現象記述であり戦略でも判定でもない（K不変・先読みなし）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
