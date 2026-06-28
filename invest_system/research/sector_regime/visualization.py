"""レジーム検知結果の可視化。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .detectors import RegimeOutput


def _setup_japanese_font() -> None:
    """Windows/macOS で日本語ラベルが表示されるようフォントを設定。"""
    for fam in ("Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP"):
        try:
            plt.rcParams["font.family"] = fam
            return
        except Exception:  # noqa: BLE001
            continue


_setup_japanese_font()


def plot_regime_on_price(
    weekly: pd.DataFrame,
    output: RegimeOutput,
    *,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """価格（累積）にレジームを色分けしたプロット。"""
    sub = weekly[weekly["sector"] == output.sector].set_index("date").sort_index()
    regime = output.regime.reindex(sub.index)

    cum = (1.0 + sub["log_return_w"].fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {0: "#2ecc71", 1: "#e74c3c", 2: "#3498db", 3: "#9b59b6"}

    prev_r = None
    seg_start = sub.index[0]
    for dt, r in regime.items():
        if prev_r is not None and r != prev_r and np.isfinite(prev_r):
            ax.axvspan(seg_start, dt, alpha=0.15, color=colors.get(int(prev_r), "#95a5a6"))
            seg_start = dt
        prev_r = r
    if prev_r is not None and np.isfinite(prev_r):
        ax.axvspan(seg_start, sub.index[-1], alpha=0.15, color=colors.get(int(prev_r), "#95a5a6"))

    ax.plot(cum.index, cum.values, color="black", linewidth=1.2, label="累積リターン")
    ax.set_title(title or f"{output.sector} — {output.method}")
    ax.set_ylabel("累積リターン指数")

    patches = [mpatches.Patch(color=colors[k], alpha=0.4, label=f"Regime {k}")
               for k in sorted(colors) if k < output.n_states]
    line_patch = mpatches.Patch(color="black", label="累積リターン")
    ax.legend(handles=[line_patch] + patches, loc="upper left")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_method_comparison_heatmap(
    ranking: pd.DataFrame,
    *,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """セクター×手法の複合スコアヒートマップ。"""
    if ranking.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "データなし", ha="center")
        return fig

    pivot = ranking.pivot_table(
        index="sector", columns="method", values="composite_score", aggfunc="max",
    )
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.5), max(6, pivot.shape[0] * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("複合スコア（セクター × 手法）")
    fig.colorbar(im, ax=ax, label="composite_score")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_score_breakdown(
    best_df: pd.DataFrame,
    *,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """セクターごとベスト手法のスコア内訳バーチャート。"""
    if best_df.empty:
        fig, ax = plt.subplots()
        return fig

    cols = ["pred_score", "quality_score", "stability_score", "econ_score"]
    labels = ["予測力", "レジーム質", "安定性", "経済的意味"]
    x = np.arange(len(best_df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(best_df) * 0.8), 5))
    for i, (col, lab) in enumerate(zip(cols, labels)):
        ax.bar(x + i * width, best_df[col].values, width, label=lab)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(
        [f"{r.sector}\n{r.method}" for r in best_df.itertuples()],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("スコア（0-1）")
    ax.set_title("ベスト手法のスコア内訳")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig