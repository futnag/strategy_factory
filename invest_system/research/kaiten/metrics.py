"""パフォーマンス集計とエクイティ曲線の描画。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def performance_report(eq: pd.DataFrame, tr: pd.DataFrame, cfg: dict) -> None:
    print("\n" + "=" * 60)
    label = cfg.get("instrument_label") or cfg.get("external_key") or ""
    title = "回転手法バックテスト  結果サマリー"
    if label:
        title = f"{title}（{label}）"
    print(f"  {title}")
    print("=" * 60)

    if eq.empty:
        print("エクイティが空です。データを確認してください。")
        return

    init = cfg["initial_capital"]
    final = eq["Equity"].iloc[-1]
    total_ret = final / init - 1

    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (final / init) ** (1 / years) - 1 if final > 0 else -1

    roll_max = eq["Equity"].cummax()
    dd = eq["Equity"] / roll_max - 1
    max_dd = dd.min()

    daily = eq["Equity"].pct_change().dropna()
    if daily.std(ddof=0) > 0:
        sharpe = daily.mean() / daily.std(ddof=0) * np.sqrt(cfg["annualization_days"])
    else:
        sharpe = float("nan")

    print(f"期間            : {eq.index[0].date()} 〜 {eq.index[-1].date()}  ({years:.2f}年)")
    print(f"初期資金        : {init:,.0f}")
    print(f"最終資産        : {final:,.0f}")
    print(f"総リターン      : {total_ret:+.2%}")
    print(f"CAGR            : {cagr:+.2%}")
    print(f"最大ドローダウン: {max_dd:+.2%}")
    print(f"Sharpe (年率)   : {sharpe:.2f}")
    print("-" * 60)

    if tr.empty:
        print("トレードが1件も発生しませんでした。")
        print("  → 指値が深すぎ/トレンド条件が厳しすぎる可能性。")
        print("     atr_mult, bb_sigma, trend_ma_period を緩めて再試行を。")
        print("=" * 60 + "\n")
        return

    n = len(tr)
    wins = tr[tr["pnl"] > 0]
    losses = tr[tr["pnl"] <= 0]
    win_rate = len(wins) / n
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"トレード回数    : {n}")
    print(f"勝率            : {win_rate:.2%}  ({len(wins)}勝 {len(losses)}敗)")
    print(f"プロフィットF   : {pf:.2f}")
    print(f"平均利益        : {wins['pnl'].mean() if len(wins) else 0.0:,.0f}")
    print(f"平均損失        : {losses['pnl'].mean() if len(losses) else 0.0:,.0f}")
    print(f"期待値/トレード : {tr['pnl'].mean():,.0f}")
    print(f"平均保有日数    : {tr['hold_days'].mean():.1f} 日")
    print("イグジット内訳  :")
    for reason, cnt in tr["reason"].value_counts().items():
        print(f"    {reason:16s}: {cnt}")
    print("=" * 60 + "\n")


def plot_equity(eq: pd.DataFrame, cfg: dict) -> None:
    if eq.empty:
        return
    import matplotlib.pyplot as plt

    roll_max = eq["Equity"].cummax()
    dd = eq["Equity"] / roll_max - 1

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(eq.index, eq["Equity"], color="#1f77b4", lw=1.4)
    ax1.axhline(cfg["initial_capital"], color="gray", ls="--", lw=0.8)
    label = cfg.get("instrument_label") or cfg.get("external_key") or "Kaiten"
    ax1.set_title(f"Equity Curve ({label})")
    ax1.set_ylabel("Equity")
    ax1.grid(alpha=0.3)

    ax2.fill_between(eq.index, dd * 100, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = cfg["plot_path"]
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[INFO] エクイティ曲線を保存: {path}")