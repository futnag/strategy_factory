"""因果調整プレミアムの精緻化（docs/43 §5 の2点）— 使い捨て・K不変・判定なし。

docs/43 の2つの限界を潰す：
  精緻化1：全部入り(60)の早期欠測 → **NaN≤3% のクリーン特性**だけで統制（A/B 含む
     全エポックで推定可能にして仕様比較を対称化）。v1 の kitchen-sink は seasonality
     45%NaN 等のため実質 2021-07 以降のみだった。
  精緻化2：backdoor 近似の粗さ → **PC のオリエンテーション**（Meek 規則）で**厳密な
     親**（Z→target）を調整集合に採用。CPDAG が向き付けできた辺／無向のままの辺を
     明示し、**調整集合が同定可能か**を報告する（同定不能なら正直にそう書く）。

すべて data/ ローカルのみ（オフライン・先読みなし）。判定（DSR）はしない。
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

from diag_causal_adjustment import (  # noqa: E402
    DISCOVERY_VARS, EPOCHS, TARGETS, cs_standardize, fama_macbeth, _epoch_slice,
)

DATA = ROOT / "data" / "phase4"


def main() -> int:
    warnings.filterwarnings("ignore")
    print("=" * 78)
    print("因果調整プレミアムの精緻化（クリーン統制＋PC厳密backdoor・K不変・判定なし）")
    print("=" * 78)

    X = pd.read_parquet(DATA / "X.parquet")
    y = pd.read_parquet(DATA / "y.parquet")["y"]
    Xz = cs_standardize(X)

    # ---- 精緻化1：クリーン統制集合（NaN≤3%）------------------------------
    nan_rate = Xz.isna().mean()
    BROAD = sorted(nan_rate[nan_rate <= 0.03].index.tolist())
    dropped = sorted(nan_rate[nan_rate > 0.03].index.tolist())
    print(f"\n[精緻化1] クリーン統制 BROAD = NaN≤3% の {len(BROAD)}特性")
    print(f"  除外({len(dropped)}・早期欠測/分散ゼロ): {dropped}")

    # ---- 精緻化2：PC 厳密 backdoor（オリエンテーション）-------------------
    print("\n[精緻化2] PC（Fisher-Z・alpha=0.01）→ 厳密な親で調整集合を同定")
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    dv = [v for v in DISCOVERY_VARS if v in Xz.columns]
    disc = pd.concat([Xz[dv], y.rename("y")], axis=1).dropna()
    if len(disc) > 40000:
        disc = disc.sample(40000, random_state=0).sort_index()
    cg = pc(disc.values, alpha=0.01, indep_test=fisherz, show_progress=False)
    names = list(dv) + ["y"]
    nodes = cg.G.get_nodes()
    g = cg.G.graph

    # 向き付け統計（有向辺数 / 無向辺数）
    directed = undirected = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = g[i, j], g[j, i]
            if a == 0 and b == 0:
                continue
            if (a == -1 and b == 1) or (a == 1 and b == -1):
                directed += 1
            else:
                undirected += 1
    print(f"  CPDAG：有向辺 {directed}・無向辺 {undirected}"
          f"（無向が多いほど調整集合は同定不能）")

    def idx(nm):
        return names.index(nm)

    def parents(nm):
        return [nodes.index(p) for p in cg.G.get_parents(nodes[idx(nm)])]

    def children(nm):
        return [nodes.index(c) for c in cg.G.get_children(nodes[idx(nm)])]

    def undirected_nb(nm):
        i = idx(nm)
        out = []
        for j in range(len(names)):
            if j == i:
                continue
            if g[i, j] == -1 and g[j, i] == -1:
                out.append(j)
        return out

    backdoor = {}
    for tg in TARGETS:
        if tg not in names:
            continue
        pa = [names[k] for k in parents(tg) if names[k] != "y"]
        ch = [names[k] for k in children(tg)]
        un = [names[k] for k in undirected_nb(tg) if names[k] != "y"]
        backdoor[tg] = pa
        print(f"\n  [{tg}]")
        print(f"    厳密な親(→{tg}・backdoor集合) : {pa if pa else '（同定なし）'}")
        print(f"    子(={tg}→・collider/媒介=除外): {ch if ch else '（なし）'}")
        print(f"    無向の隣接(向き不明)          : {un if un else '（なし）'}")

    # ---- プレミアム：素朴 / 厳密因果 / クリーン統制(BROAD) ---------------
    print("\n" + "=" * 78)
    print("プレミアム（×100・NW-t）：素朴 / 厳密因果backdoor / クリーン統制(BROAD)")
    print("=" * 78)
    series = {}
    for tg in TARGETS:
        m_n, t_n, s_n = fama_macbeth(y, Xz[[tg]], tg)
        adj = [tg] + [c for c in backdoor.get(tg, []) if c in Xz.columns]
        m_c, t_c, s_c = fama_macbeth(y, Xz[adj], tg)
        bro = sorted(set([tg] + BROAD))
        m_b, t_b, s_b = fama_macbeth(y, Xz[bro], tg)
        series[tg] = {"素朴": s_n, "厳密因果": s_c, "クリーン統制": s_b}
        print(f"\n[{tg}]")
        print(f"   素朴(単変量)         : {m_n*100:+.3f}  t={t_n:+.2f}  "
              f"(月数{len(s_n)})")
        print(f"   厳密因果({len(adj)-1}統制)      : {m_c*100:+.3f}  t={t_c:+.2f}  "
              f"(月数{len(s_c)})")
        print(f"   クリーン統制({len(bro)-1}統制) : {m_b*100:+.3f}  t={t_b:+.2f}  "
              f"(月数{len(s_b)})")

    # ---- エポック別安定性（クリーン統制は全エポックで推定可のはず）-------
    print("\n" + "=" * 78)
    print("エポック別プレミアム（×100）と符号一致・CV（精緻化後）")
    print("=" * 78)
    for tg in TARGETS:
        print(f"\n[{tg}]")
        print(f"  {'spec':14}" + "".join(f"{e[0][:8]:>10}" for e in EPOCHS)
              + f"{'符号一致':>8}{'CV':>7}")
        for spec_name, s in series[tg].items():
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
            consist = f"{int(abs(sum(signs)))}/{len(signs)}" if signs else "—"
            cv = (np.std(vals) / abs(np.mean(vals))
                  if vals and abs(np.mean(vals)) > 1e-9 else np.nan)
            print(f"  {spec_name:14}{cells}{consist:>8}{cv:>7.2f}")

    print("\n精緻化の要点：")
    print("  1) クリーン統制は全4エポックで推定可（v1の全部入りはC/Dのみ＝非対称を解消）。")
    print("  2) PC厳密backdoorは同定可能な親のみ採用。無向辺が多ければ調整集合は同定不能。")
    print("\n本診断は推定方法の現象記述であり戦略でも判定でもない（K不変・先読みなし）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
