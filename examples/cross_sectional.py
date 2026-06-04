"""クロスセクション・ファクター投資（柱C）：複数crypto銘柄の相対価値を検証。

単一資産の「方向」予測（最難問・エッジ無し）から、資料群の本丸である
クロスセクション相対価値へ転換する。bitbank の主要 JPY ペアでパネルを作り、
各時点で銘柄を横断ランクして long-short（ドルニュートラル）ファクターを構成：
  - モメンタム（過去Lリターン上位ロング/下位ショート）
  - 短期リバーサル（直近R下落をロング）
  - 低ボラ（低ボラをロング）
各ファクターの Sharpe を求め、試したファクター数で DSR を補正（多重検定）。

実行（リポジトリ root から）: python examples/cross_sectional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources.bitbank import fetch_candlesticks  # noqa: E402
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


def factor_returns(signal: pd.DataFrame, fwd: pd.DataFrame,
                   q: float = 0.33, min_assets: int = 6) -> pd.Series:
    """各日に signal 上位qをロング/下位qをショート（ドルニュートラル, 等加重）。"""
    rank = signal.rank(axis=1, pct=True)
    n = signal.notna().sum(axis=1)
    long_ret = fwd.where(rank >= 1 - q).mean(axis=1)
    short_ret = fwd.where(rank <= q).mean(axis=1)
    fr = (long_ret - short_ret).where(n >= min_assets)
    return fr.replace([np.inf, -np.inf], np.nan).dropna()


def main() -> None:
    print(f"bitbank から日次足を取得中 ... {len(PAIRS)}銘柄 × {YEARS[0]}–{YEARS[-1]}")
    panel = fetch_panel(PAIRS, YEARS)
    ret = panel.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    print(f"パネル: {panel.shape[0]}日 × {panel.shape[1]}銘柄  "
          f"{panel.index[0].date()}〜{panel.index[-1].date()}  "
          f"平均有効銘柄数 {panel.notna().sum(axis=1).mean():.1f}\n")

    factors = {
        "モメンタム15": panel.pct_change(15),
        "モメンタム30": panel.pct_change(30),
        "モメンタム60": panel.pct_change(60),
        "リバーサル2": -panel.pct_change(2),
        "リバーサル5": -panel.pct_change(5),
        "低ボラ20": -ret.rolling(20).std(),
    }
    reg = TrialRegistry(":memory:")
    scope = "xsection_crypto"
    rows, best = [], {"sr": -np.inf, "uuid": None, "name": None}
    for name, sig in factors.items():
        fr = factor_returns(sig, fwd)
        sr = float(fr.mean() / fr.std()) if fr.std() > 0 else 0.0
        rows.append((name, sr * np.sqrt(365), sr, len(fr)))
        tid = reg.preregister(scope=scope, hypothesis=f"cross-sectional {name} long-short",
                              economic_rationale="relative-value premium across crypto assets")
        reg.record_result(tid, sharpe=sr, n_obs=len(fr), skew=0.0, kurt=3.0)
        if sr > best["sr"]:
            best.update(sr=sr, uuid=tid, name=name)

    print("=== クロスセクション・ファクター（long-short, ドルニュートラル）===")
    print(f"{'ファクター':16}{'年率Sharpe':>12}{'日次Sharpe':>12}{'日数':>8}")
    for name, ann, sr, n in rows:
        print(f"{name:16}{ann:>12.2f}{sr:>12.4f}{n:>8}")

    dsr = reg.deflated_sharpe(best["uuid"])
    print(f"\n試行数 K = {reg.trial_count(scope)}（試したファクター数）")
    print(f"最良ファクター: {best['name']}  → DSR（多重検定補正後）: {dsr:.3f}")
    if dsr >= 0.95:
        print("→ 有意な相対価値プレミアムの兆候。次は因果フィルタで交絡/コライダーを精査し、"
              "\n  HRP/NCO で頑健に合成、感度分析・別期間で追検証（過学習に警戒）。")
    else:
        print("→ DSR<0.95＝多重検定補正後は有意でない。生の最良Sharpeに釣られない正直な評価。"
              "\n  次は別ホライズン/ウェイト方式（ランク加重）/ボラ調整/より多銘柄で継続探索。")


if __name__ == "__main__":
    main()
