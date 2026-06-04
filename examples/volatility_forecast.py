"""ボラティリティ予測（パスB）：方向は予測不能でもボラは予測可能。

López de Prado の「価格（方向）予測はMLの最も信頼性が低い用途」を踏まえ、問題を
方向予測からボラ予測へ切り替える。実 BTC/JPY で purged CV の OOS R² を比較：
  - 将来実現ボラ（log）の予測 R²  … ボラクラスタリングにより高いはず
  - 将来リターン（方向）の予測 R²  … ほぼ 0（予測不能）
さらに「現在ボラ＝将来ボラ」の素朴な持続性ベンチマークと ML を比較する。

実行（リポジトリ root から）: python examples/volatility_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from invest_system.backtest.cv_score import purged_cv_predict  # noqa: E402
from invest_system.data.sources.bitbank import fetch_candlesticks  # noqa: E402
from invest_system.features.microstructure import (  # noqa: E402
    garman_klass_vol,
    parkinson_vol,
    rsi,
    vpin,
)
from invest_system.labeling.triple_barrier import get_vertical_barriers  # noqa: E402

YEARS = ("2021", "2022", "2023", "2024", "2025", "2026")
H = 6   # 予測ホライズン（4時間足×6 = 24時間先）


def _rf():
    return RandomForestRegressor(n_estimators=120, max_depth=6,
                                 min_samples_leaf=20, random_state=0, n_jobs=-1)


def main() -> None:
    print(f"bitbank から 4時間足 BTC/JPY を取得中 ... {YEARS}")
    cd = fetch_candlesticks("btc_jpy", "4hour", YEARS)
    o, h, l, c, v = cd["open"], cd["high"], cd["low"], cd["close"], cd["volume"]
    ret = c.pct_change(fill_method=None)
    print(f"取得: {len(c)} 本  {c.index[0].date()} 〜 {c.index[-1].date()}\n")

    rv_h = ret.rolling(H).std()
    feats = pd.DataFrame({                                   # 全て時刻 t までの情報（先読み無し）
        "rv5": ret.rolling(5).std(),
        "rv10": ret.rolling(10).std(),
        "rv_h": rv_h,
        "park": parkinson_vol(h, l, 10),
        "gk": garman_klass_vol(o, h, l, c, 10),
        "vpin": vpin(c, v, 30),
        "absret": ret.abs(),
        "ret": ret,                                         # レバレッジ効果（下落→高ボラ）
        "rsi": rsi(c, 14),
        "logvol": np.log(v.replace(0, np.nan)),
    })
    df = feats.copy()
    df["t_vol"] = np.log(rv_h.shift(-H))                    # 将来H本の実現ボラ(log)
    df["t_ret"] = c.pct_change(H).shift(-H)                 # 将来H本のリターン（方向）
    df["cur_logvol"] = np.log(rv_h)                         # 持続性ベンチの予測子
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    vb = get_vertical_barriers(c, df.index, num_bars=H)     # t1 = H本先（パージ用）
    df = df.loc[vb.index]
    X, t1 = df[feats.columns], vb

    oos_vol = purged_cv_predict(X, df["t_vol"], t1, _rf)
    oos_ret = purged_cv_predict(X, df["t_ret"], t1, _rf)
    # 較正済み単一特徴ベンチ：現在ボラのみで将来ボラを予測（公平な持続性比較）
    oos_persist = purged_cv_predict(df[["cur_logvol"]], df["t_vol"], t1, LinearRegression)
    r2_vol = r2_score(df["t_vol"], oos_vol.to_numpy())
    r2_persist = r2_score(df["t_vol"], oos_persist.to_numpy())
    r2_ret = r2_score(df["t_ret"], oos_ret.to_numpy())

    print(f"=== 予測可能性の対比（実BTC/JPY, purged CV OOS R², {H*4}h 先）===")
    print(f"観測数 {len(df)}")
    print(f"{'対象':28}{'OOS R²':>10}")
    print(f"{'将来ボラ（ML・全特徴）':24}{r2_vol:>12.3f}")
    print(f"{'将来ボラ（現在ボラ1特徴）':24}{r2_persist:>12.3f}")
    print(f"{'将来リターン/方向（ML）':24}{r2_ret:>12.3f}")
    print()
    if r2_vol > 0.2 and r2_ret < 0.05:
        print("→ ボラは明確に予測可能（R²>0）／方向はほぼ予測不能（R²≈0）。")
        print("  López de Prado の『方向でなくボラ・サイジング・リスクにMLを使え』を実データで確認。")
    else:
        print("→ 結果を額面通り解釈（過学習・期間依存に注意）。")
    print("\n応用：動的トリプルバリア幅、ボラ・ターゲティング（リスク一定化）、"
          "ポジションサイジング、レジーム検知。perp/オプションがあればボラ自体の売買も。")


if __name__ == "__main__":
    main()
