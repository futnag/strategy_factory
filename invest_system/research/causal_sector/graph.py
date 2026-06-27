"""PCMCI+ 因果グラフ学習と collider 同定。

tigramite の ParCorr + PCMCI+ を使用。RPCMCI は条件付き独立性の
ロバスト版として ParCorr を既定とし、外れ値は winsorize で緩和。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import SectorProfile, collider_candidates


@dataclass
class CausalGraphResult:
    """PCMCI+ 学習結果。"""
    var_names: list[str]
    val_matrix: np.ndarray
    graph: np.ndarray
    tau_max: int
    colliders: list[str] = field(default_factory=list)
    adjustment_set: dict[str, list[str]] = field(default_factory=dict)
    parents: dict[str, list[tuple[str, int, float]]] = field(default_factory=dict)


def _winsorize_df(df: pd.DataFrame, lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    q = df.quantile([lo, hi])
    return df.clip(lower=q.loc[lo], upper=q.loc[hi], axis=1)


def learn_pcmci_graph(panel: pd.DataFrame, *,
                      treatment: str = "VALUE",
                      outcome: str = "RET",
                      tau_max: int = 5,
                      pc_alpha: float = 0.05,
                      min_obs: int = 200) -> CausalGraphResult:
    """PCMCI+ で因果グラフを学習。

    Parameters
    ----------
    panel : 日次パネル（内生＋外生）。欠損行は dropna。
    treatment, outcome : VALUE→RET の因果エッジ監視対象。
    """
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    df = panel.dropna()
    if len(df) < min_obs:
        raise ValueError(f"insufficient obs: {len(df)} < {min_obs}")

    # 内生変数を先に、外生ドライバを後に並べる（解釈しやすさ）
    endogenous = [c for c in ("VALUE", "RET", "MOM", "VOL", "SHORT_RATIO")
                  if c in df.columns]
    exogenous = [c for c in df.columns if c not in endogenous]
    var_names = endogenous + exogenous

    Z = _winsorize_df(df[var_names])
    Z = (Z - Z.mean()) / Z.std()
    Z = Z.replace([np.inf, -np.inf], np.nan).dropna()
    if len(Z) < min_obs:
        raise ValueError(f"insufficient obs after standardize: {len(Z)}")

    dataframe = pp.DataFrame(Z.values, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)
    val, graph = res["val_matrix"], res["graph"]

    name = {i: v for i, v in enumerate(var_names)}
    parents: dict[str, list[tuple[str, int, float]]] = {}
    for j, vn in enumerate(var_names):
        p_list = []
        for i in range(len(var_names)):
            for lag in range(0, tau_max + 1):
                if lag == 0 and i == j:
                    continue
                g = graph[i, j, lag]
                if g in ("-->", "o->"):
                    p_list.append((name[i], lag, float(val[i, j, lag])))
        p_list.sort(key=lambda t: -abs(t[2]))
        parents[vn] = p_list

    colliders = _identify_colliders_from_graph(
        var_names, graph, val, tau_max, treatment, outcome)
    adj = _backdoor_adjustment_set(parents, treatment, colliders, var_names, outcome)
    # collider は VALUE→RET の調整集合から必ず除外（mirage 防止）
    adj["VALUE_to_RET"] = [v for v in adj.get("VALUE_to_RET", []) if v not in colliders]

    return CausalGraphResult(
        var_names=var_names, val_matrix=val, graph=graph, tau_max=tau_max,
        colliders=colliders, adjustment_set=adj, parents=parents,
    )


def _identify_colliders_from_graph(var_names, graph, val, tau_max,
                                   treatment: str, outcome: str) -> list[str]:
    """T→C かつ Y→C の変数 C を collider として同定（lag0 のみ）。"""
    idx = {v: i for i, v in enumerate(var_names)}
    ti, yi = idx.get(treatment), idx.get(outcome)
    if ti is None or yi is None:
        return []
    colliders = []
    for c, ci in idx.items():
        if c in (treatment, outcome):
            continue
        t_to_c = graph[ti, ci, 0] in ("-->", "o->") or any(
            graph[ti, ci, lag] in ("-->", "o->") for lag in range(1, tau_max + 1))
        y_to_c = graph[yi, ci, 0] in ("-->", "o->") or any(
            graph[yi, ci, lag] in ("-->", "o->") for lag in range(1, tau_max + 1))
        c_to_t = graph[ci, ti, 0] in ("-->", "o->")
        c_to_y = graph[ci, yi, 0] in ("-->", "o->")
        # 両方向から指向＝合流点候補（子への矢印が T,Y 双方から）
        if (t_to_c and y_to_c) or (c_to_t and c_to_y and t_to_c):
            colliders.append(c)
    return sorted(set(colliders))


def _backdoor_adjustment_set(parents: dict, treatment: str,
                             colliders: list[str],
                             var_names: list[str],
                             outcome: str = "RET") -> dict[str, list[str]]:
    """各エンドポイントの調整集合＝親から collider を除外。"""
    coll_set = set(colliders)
    adj = {}
    for vn in var_names:
        pnames = [p[0] for p in parents.get(vn, []) if p[1] >= 0]
        adj[vn] = [p for p in pnames if p not in coll_set and p != vn]
    # VALUE→RET の backdoor: VALUE の親 ∪ RET の親（collider 除く）
    value_parents = [p for p in adj.get(treatment, []) if p != outcome]
    ret_parents = [p for p in adj.get(outcome, []) if p != treatment]
    adj["VALUE_to_RET"] = sorted(set(value_parents + ret_parents) - coll_set)
    return adj


def identify_colliders(result: CausalGraphResult,
                       profile: Optional[SectorProfile] = None) -> list[str]:
    """グラフ同定＋事前 watch リストの和集合。"""
    watch = []
    if profile is not None:
        watch = collider_candidates(result.var_names, profile)
    return sorted(set(result.colliders) | set(watch))


def graph_summary(result: CausalGraphResult, *,
                  treatment: str = "VALUE",
                  outcome: str = "RET") -> pd.DataFrame:
    """因果エッジ一覧（DataFrame）。"""
    rows = []
    name = {i: v for i, v in enumerate(result.var_names)}
    for j, tgt in enumerate(result.var_names):
        for i, src in enumerate(result.var_names):
            for lag in range(0, result.tau_max + 1):
                if lag == 0 and i == j:
                    continue
                g = result.graph[i, j, lag]
                if g in ("-->", "o->", "<--", "<-o", "o-o", "---"):
                    rows.append({
                        "source": name[i], "target": name[j], "lag": lag,
                        "link": g, "mci": float(result.val_matrix[i, j, lag]),
                        "is_value_edge": src == treatment and tgt == outcome,
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mci", key=abs, ascending=False)
    return df


def plot_causal_graph(result: CausalGraphResult, path: str, *,
                      treatment: str = "VALUE", outcome: str = "RET",
                      title: str = "") -> None:
    """networkx で lag=1 の主要エッジを可視化（保存）。"""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    summ = graph_summary(result)
    if summ.empty:
        return
    edges = summ[(summ["lag"] >= 1) & (summ["lag"] <= 2)]
    edges = edges[edges["link"].isin(("-->", "o->"))]
    for _, r in edges.head(20).iterrows():
        G.add_edge(r["source"], r["target"],
                   weight=abs(r["mci"]), mci=r["mci"])

    pos = nx.spring_layout(G, seed=42)
    colors = []
    for n in G.nodes():
        if n == treatment:
            colors.append("#e74c3c")
        elif n == outcome:
            colors.append("#3498db")
        elif n in result.colliders:
            colors.append("#f39c12")
        else:
            colors.append("#95a5a6")

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors,
            node_size=1200, font_size=8, arrows=True,
            edge_color="#7f8c8d", width=1.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)