"""マルチファクター・クロスセクション（柱C capstone）。

クロスセクションの弱い単独ファクター（Sharpe~0.5）を、柱Cツールで強化・合成する：
  L4 因果フィルタ … 各ファクターが将来リターンの「原因」か「コライダー」かを判定
  ファクター合成   … 採用ファクターを横断zスコアで合成（分散効果）
  L9 HRP          … ファクター・スリーブをリスクベースで合成（代替案）
  L8 DSR          … 試行（ファクター数＋合成法）を補正した正直な有意性評価

実 bitbank 15銘柄パネルで、合成が単独ファクターを上回り DSR を押し上げるかを検証。

実行（リポジトリ root から）: python examples/multifactor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources.bitbank import fetch_candlesticks  # noqa: E402
from invest_system.features.causal import classify_features  # noqa: E402
from invest_system.portfolio.allocation import hrp_weights  # noqa: E402
from invest_system.validation.registry import TrialRegistry  # noqa: E402

PAIRS = ["btc_jpy", "eth_jpy", "xrp_jpy", "ltc_jpy", "bcc_jpy", "mona_jpy",
         "xlm_jpy", "qtum_jpy", "bat_jpy", "link_jpy", "doge_jpy", "dot_jpy",
         "ada_jpy", "matic_jpy", "avax_jpy"]
YEARS = ["2022", "2023", "2024", "2025", "2026"]


def fetch_panel(pairs, years) -> pd.DataFrame:
    closes = {}
    for p in pairs:
        frames = []
        for y in years:
            try:
                frames.append(fetch_candlesticks(p, "1day", [y], pause=0.1))
            except Exception:
                pass
        if frames:
            df = pd.concat(frames).sort_index()
            closes[p] = df[~df.index.duplicated(keep="first")]["close"]
    return pd.DataFrame(closes).sort_index()


def factor_returns(signal, fwd, q=0.33, min_assets=6) -> pd.Series:
    rank = signal.rank(axis=1, pct=True)
    n = signal.notna().sum(axis=1)
    fr = (fwd.where(rank >= 1 - q).mean(axis=1)
          - fwd.where(rank <= q).mean(axis=1)).where(n >= min_assets)
    return fr.replace([np.inf, -np.inf], np.nan).dropna()


def _zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def ann(sr):
    return sr * np.sqrt(365)


def main() -> None:
    print(f"bitbank から日次足を取得中 ... {len(PAIRS)}銘柄 × {YEARS[0]}–{YEARS[-1]}")
    panel = fetch_panel(PAIRS, YEARS)
    ret = panel.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    print(f"パネル {panel.shape[0]}日 × {panel.shape[1]}銘柄 "
          f"{panel.index[0].date()}〜{panel.index[-1].date()}\n")

    factors = {
        "mom30": panel.pct_change(30),
        "mom60": panel.pct_change(60),
        "rev3": -panel.pct_change(3),
        "lowvol": -ret.rolling(20).std(),
    }

    # 1) 因果ガバナンス：プールして各ファクターを cause/effect 分類
    pooled = pd.DataFrame({n: s.stack() for n, s in factors.items()})
    pooled["fut"] = fwd.stack()
    pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna()
    cls = classify_features(pooled[list(factors)], pooled["fut"], threshold=-0.05)
    keep = [f for f in factors if cls.loc[f, "role"] == "cause"]
    print("=== [L4] 因果ガバナンス（将来リターンに対する各ファクターの役割）===")
    print(cls.sort_values("score", ascending=False))
    print(f"採用ファクター: {keep}\n")

    # 2) 個別ファクター収益
    frets = pd.DataFrame({f: factor_returns(factors[f], fwd) for f in keep}).dropna()
    sr = {f: float(frets[f].mean() / frets[f].std()) for f in keep}

    # 3) 合成：横断zスコア等加重コンポジット
    comp_sig = sum(_zscore(factors[f]) for f in keep)
    comp = factor_returns(comp_sig, fwd).reindex(frets.index).dropna()
    sr_comp = float(comp.mean() / comp.std())

    # 4) HRP スリーブ合成（リスクベース、リターン符号に非依存）
    hrp_w = hrp_weights(frets.cov())
    hrp_ret = (frets * hrp_w).sum(axis=1)
    sr_hrp = float(hrp_ret.mean() / hrp_ret.std())

    # 5) DSR（試行＝個別ファクター＋合成2法）
    reg = TrialRegistry(":memory:")
    scope = "multifactor"
    strategies = {**{f: (sr[f], len(frets)) for f in keep},
                  "zコンポジット": (sr_comp, len(comp)),
                  "HRP合成": (sr_hrp, len(hrp_ret))}
    best = {"sr": -np.inf, "uuid": None, "name": None}
    for name, (s, nobs) in strategies.items():
        tid = reg.preregister(scope=scope, hypothesis=f"cross-sectional {name}",
                              economic_rationale="multi-factor relative-value premium")
        reg.record_result(tid, sharpe=s, n_obs=nobs, skew=0.0, kurt=3.0)
        if s > best["sr"]:
            best.update(sr=s, uuid=tid, name=name)
    dsr = reg.deflated_sharpe(best["uuid"])

    print("=== マルチファクター合成（long-short, ドルニュートラル）===")
    print(f"{'戦略':16}{'年率Sharpe':>12}")
    for f in keep:
        print(f"{f:16}{ann(sr[f]):>12.2f}")
    print(f"{'zコンポジット':14}{ann(sr_comp):>12.2f}")
    print(f"{'HRP合成':16}{ann(sr_hrp):>12.2f}")
    print(f"\n試行数 K = {reg.trial_count(scope)}  最良: {best['name']}  "
          f"→ DSR（多重検定補正後）: {dsr:.3f}")
    if dsr >= 0.95:
        print("→ 有意な相対価値プレミアム。感度分析・別期間・取引コスト控除で追検証へ。")
    else:
        print("→ DSR<0.95＝補正後は未有意。合成でも閾値未達。正直な評価。"
              "\n  次：ファクター拡充（キャリー/ボラ/流動性）、銘柄拡大、週次リバランス、コスト現実化。")


if __name__ == "__main__":
    main()
