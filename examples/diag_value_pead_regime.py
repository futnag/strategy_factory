"""value/PEAD プレミアムのレジーム別推定（使い捨て診断・K不変・オフライン）。

docs/27 §8 の限界（日次キャッシュに book/market が無く value LS を再構築できず）を、
**月次 GKX 設計行列** data/phase4/X.parquet（book_to_market・earnings_yield・
forecast_revision・sue を収録）＋ y.parquet（1か月フォワード超過リターン）で解消する。

各月 t で因子値の上位/下位 quintile の y を差し引いてロングショート・プレミアムを作り、
docs/27 の主要レジーム転換（2020-03 COVID, 2022-09 利上げ/value復権）でエポック分割して
Sharpe を比較。＝「value/PEAD の優劣がレジームでどう反転したか」を月次横断面で定量確認。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "phase4"
EPOCHS = [("A 平穏ブル", "2016-07", "2018-09"),
          ("B 後期→COVID", "2018-10", "2020-03"),
          ("C コロナ後リフレ", "2020-04", "2022-08"),
          ("D 利上げ/value復権", "2022-09", "2030-12")]
FACTORS = {"book_to_market": "value(B/M)", "earnings_yield": "value(E/P)",
           "forecast_revision": "PEAD(予想改訂)", "sue_recent": "PEAD(SUE直近)"}


def factor_ls(X, y, fname, q=0.2, min_n=30):
    """各月 t：因子 fname 上位q − 下位q の y（フォワード超過）平均＝LSプレミアム。"""
    d = pd.DataFrame({"f": X[fname], "y": y}).dropna()

    def ls(g):
        if len(g) < min_n:
            return np.nan
        k = max(1, int(len(g) * q))
        s = g["y"].to_numpy()[np.argsort(g["f"].to_numpy())]
        return float(s[-k:].mean() - s[:k].mean())
    return d.groupby(level="date").apply(ls).dropna()


def ann_sharpe(s, ann=12):
    s = s.dropna()
    sd = s.std(ddof=1)
    return float(s.mean() / sd * np.sqrt(ann)) if len(s) >= 6 and sd > 0 else float("nan")


def main() -> int:
    X = pd.read_parquet(DATA / "X.parquet")
    y = pd.read_parquet(DATA / "y.parquet")["y"]
    print("=" * 84)
    print("value/PEAD プレミアム × レジーム（月次GKX行列・1MフォワードLS・年率Sharpe）")
    print("=" * 84)
    print(f"行列 {X.shape} / 期間 {X.index.get_level_values('date').min():%Y-%m}"
          f"〜{X.index.get_level_values('date').max():%Y-%m}\n")

    series = {}
    for f in FACTORS:
        if f in X.columns:
            series[f] = factor_ls(X, y, f)

    # --- 全期間＋エポック別 Sharpe ---
    hdr = f"{'因子':<18}{'全期間':>8}" + "".join(f"{e[0][:10]:>13}" for e in EPOCHS)
    print(hdr)
    for f, lab in FACTORS.items():
        if f not in series:
            continue
        s = series[f]
        row = f"{lab:<18}{ann_sharpe(s):>+8.2f}"
        for _, a, b in EPOCHS:
            seg = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
            sh = ann_sharpe(seg)
            row += f"{(f'{sh:+.2f}' if np.isfinite(sh) else '—'):>13}"
        print(row)

    # --- value(B/M) の pre/post 2020・2022 ---
    print("\n--- value(B/M) の構造転換前後（年率Sharpe）---")
    bm = series.get("book_to_market")
    if bm is not None:
        for cut in ("2020-04", "2022-09"):
            pre = ann_sharpe(bm[bm.index < pd.Timestamp(cut)])
            post = ann_sharpe(bm[bm.index >= pd.Timestamp(cut)])
            print(f"  {cut} 前 {pre:+.2f} → 後 {post:+.2f}")

    # --- value が正に転じた月（12Mローリング） ---
    if bm is not None:
        roll = bm.rolling(12).mean() / bm.rolling(12).std(ddof=1) * np.sqrt(12)
        pos = roll[(roll > 0) & (roll.shift(1) <= 0)].index
        print("\n  value(B/M) 12MローリングSharpeが正転した月:",
              "  ".join(f"{d:%Y-%m}" for d in pos) or "—")
    print("\n※ docs/27 のレジーム転換（2020-03 COVID, 2022-09 利上げ/value復権）と整合するかを確認。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
