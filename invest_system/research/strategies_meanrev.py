"""柱D 戦略：時系列・統計的裁定（mean reversion / stat-arb）。Ernie Chan 補完・KB §11。

既存 strategy.py の素朴 PairsStrategy を昇格：共和分ゲート（CADF）＋ AsOf 動的ヘッジ
（rolling-OLS / Kalman）で、§6.4 の pairs(DSR 0.02) 失敗要因（共和分未検定・固定ヘッジ比）
を修正する。全戦略は Strategy 契約（target_weights(asof)）＝判定は judge_grid に載る。
β・z は ≤t のみで推定＝先読みなし（DP12）。状態を持たず history から毎回再計算＝PIT 安全。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeseries import (
    KalmanHedge,
    cadf,
    half_life,
    hedge_ratio_ols,
    johansen,
    spread_series,
)
from .data_view import AsOf
from .strategy import Strategy


def _norm_gross(raw: pd.Series) -> pd.Series:
    """グロス1（Σ|w|=1）に正規化。全0/異常は空 Series。"""
    g = float(raw.abs().sum())
    if g == 0 or not np.isfinite(g):
        return pd.Series(dtype="float64")
    return (raw / g).astype("float64")


class CointegratedPairs(Strategy):
    """共和分ゲート＋動的ヘッジのペア平均回帰（PairsStrategy 昇格版）。

    各 t で ≤t 直近 lookback により β（ヘッジ比）と z を推定。coint_gate なら CADF の
    p≤cadf_max_p（＋任意で half_life≤max_half_life）を満たす時だけ建玉。z>entry は割高
    →スプレッド売り（a 売り/b 買い）、z<−entry は逆、|z|<entry は建てない。建玉は β加重
    [±P_a, ∓β·P_b] をグロス1正規化（ヘッジ比でマーケット中立、a≈β·b の時ほぼダラー中立）。
    exit はステートフル・ヒステリシス変種用に保持（本ステートレス版は entry のみで判定）。
    """

    def __init__(self, a, b, lookback: int = 60, entry: float = 1.5,
                 exit: float = 0.5, method: str = "rolling_ols",
                 coint_gate: bool = True, cadf_max_p: float = 0.05,
                 max_half_life=None, name: str | None = None):
        self.a, self.b = str(a), str(b)
        self.lookback = int(lookback)
        self.entry = float(entry)
        self.exit = float(exit)
        self.method = str(method)
        self.coint_gate = bool(coint_gate)
        self.cadf_max_p = float(cadf_max_p)
        self.max_half_life = max_half_life
        self.name = name or (f"coint_pairs({self.a}-{self.b},lb={lookback},"
                             f"e={entry:g},{method})")
        self.params = {"a": self.a, "b": self.b, "lookback": lookback,
                       "entry": entry, "exit": exit, "method": method,
                       "coint_gate": coint_gate, "cadf_max_p": cadf_max_p,
                       "max_half_life": max_half_life}

    def target_weights(self, asof: AsOf) -> pd.Series:
        cl = asof.frame("close")
        if self.a not in cl.columns or self.b not in cl.columns:
            return pd.Series(dtype="float64")
        sub = cl[[self.a, self.b]].dropna()
        if len(sub) < self.lookback + 2:
            return pd.Series(dtype="float64")
        win = sub.iloc[-self.lookback:]
        ya, xb = win[self.a], win[self.b]

        # --- 共和分ゲート（PIT：直近窓のみ）---
        if self.coint_gate:
            beta_g, p = cadf(ya, xb)
            if not np.isfinite(p) or p > self.cadf_max_p:
                return pd.Series(dtype="float64")
            if self.max_half_life is not None:
                hl = half_life(spread_series(ya, xb, beta_g))
                if not (hl <= float(self.max_half_life)):
                    return pd.Series(dtype="float64")

        # --- β と z（≤t のみ）---
        if self.method == "kalman":
            kf = KalmanHedge().filter(sub[self.b], sub[self.a])   # x=b, y=a
            beta = float(kf["beta"].iloc[-1])
            z = float(kf["z"].iloc[-1])
        else:                                                     # rolling_ols
            beta = hedge_ratio_ols(ya, xb)
            sp = spread_series(ya, xb, beta)
            sd = float(sp.std(ddof=0))
            if sd == 0 or not np.isfinite(sd):
                return pd.Series(dtype="float64")
            z = float((sp.iloc[-1] - sp.mean()) / sd)

        if not np.isfinite(beta) or beta <= 0 or not np.isfinite(z):
            return pd.Series(dtype="float64")        # 逆相関/異常は建玉せず

        if abs(z) < self.entry:
            return pd.Series(dtype="float64")        # 乖離不足＝フラット
        pa, pb = float(sub[self.a].iloc[-1]), float(sub[self.b].iloc[-1])
        sign = -1.0 if z > 0 else 1.0                # z>0=割高→スプレッド売り
        raw = pd.Series({self.a: sign * pa, self.b: -sign * beta * pb})
        return _norm_gross(raw)


class JohansenBasket(Strategy):
    """3資産以上の Johansen 共和分バスケット。最強固有ベクトル方向に平均回帰建玉。

    各 t で ≤t 直近 lookback の Johansen 検定。共和分関係（trace>crit95）が無ければ建てない。
    スプレッド = prices·w*（w*=最強固有ベクトル）の z で逆張り。建玉は w*×価格のドル換算を
    グロス1正規化。KB §11.3。
    """

    def __init__(self, codes, lookback: int = 120, entry: float = 1.5,
                 exit: float = 0.5, name: str | None = None):
        self.codes = [str(c) for c in codes]
        self.lookback = int(lookback)
        self.entry = float(entry)
        self.exit = float(exit)
        self.name = name or (f"johansen({'+'.join(self.codes)},lb={lookback},"
                             f"e={entry:g})")
        self.params = {"codes": self.codes, "lookback": lookback,
                       "entry": entry, "exit": exit}

    def target_weights(self, asof: AsOf) -> pd.Series:
        cl = asof.frame("close")
        cols = [c for c in self.codes if c in cl.columns]
        if len(cols) < 2:
            return pd.Series(dtype="float64")
        sub = cl[cols].dropna()
        if len(sub) < self.lookback + 2:
            return pd.Series(dtype="float64")
        win = sub.iloc[-self.lookback:]
        try:
            jr = johansen(win)
        except Exception:                            # noqa: BLE001  特異/数値失敗
            return pd.Series(dtype="float64")
        if jr.n_relations < 1:
            return pd.Series(dtype="float64")
        w = np.asarray(jr.strongest, dtype=float)
        spread = win.to_numpy() @ w
        sd = float(np.std(spread, ddof=0))
        if sd == 0 or not np.isfinite(sd):
            return pd.Series(dtype="float64")
        z = float((spread[-1] - spread.mean()) / sd)
        if abs(z) < self.entry:
            return pd.Series(dtype="float64")
        sign = -1.0 if z > 0 else 1.0
        pl = sub[cols].iloc[-1].to_numpy(dtype=float)
        return _norm_gross(pd.Series(sign * w * pl, index=cols))


class LinearMeanReversion(Strategy):
    """単一系列（指数/ETF/スプレッド）の線形スケーリング平均回帰（Chan 連続版・KB §11.5）。

    z=(price−rolling_mean)/rolling_std を ≤t で算出、weight=clip(−z/scale, −cap, +cap)。
    z<0（割安）→ロング、z>0（割高）→ショート。在庫を z に連続的に反比例させる。
    """

    def __init__(self, code, lookback: int = 60, scale: float = 2.0,
                 cap: float = 1.0, name: str | None = None):
        self.code = str(code)
        self.lookback = int(lookback)
        self.scale = float(scale)
        self.cap = float(cap)
        self.name = name or f"linrev({self.code},lb={lookback},sc={scale:g})"
        self.params = {"code": self.code, "lookback": lookback,
                       "scale": scale, "cap": cap}

    def target_weights(self, asof: AsOf) -> pd.Series:
        cl = asof.frame("close")
        if self.code not in cl.columns:
            return pd.Series(dtype="float64")
        s = cl[self.code].dropna()
        if len(s) < self.lookback + 1:
            return pd.Series(dtype="float64")
        win = s.iloc[-self.lookback:]
        sd = float(win.std(ddof=0))
        if sd == 0 or not np.isfinite(sd):
            return pd.Series(dtype="float64")
        z = float((s.iloc[-1] - win.mean()) / sd)
        w = float(np.clip(-z / self.scale, -self.cap, self.cap))
        if w == 0:
            return pd.Series(dtype="float64")
        return pd.Series({self.code: w}, dtype="float64")
