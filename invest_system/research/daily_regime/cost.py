"""daily_regime: レジーム依存コスト（§6.4・Amihud→impact・保守側固定＋感応度）— step 4。

単一銘柄はマーケットインパクト支配で、それは検知対象の低流動性レジームそのもの。フラットコストは
フィルタが狙う低流動局面でだけコストを過小評価し判定をフィルタ有利に歪める。Amihud を各銘柄の
trailing 中央値で正規化し、平常超過分を impact として base に上乗せ（≥ base＝保守）。

impact_coef は**保守側固定・非最適化**。感応度は impact_coef を振って「結論が係数のドライバーで
ないこと」を確認する（最初から本物のレジーム依存コストで組む＝後差し替えの交絡を作らない）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regime_cost_panel(amihud: pd.DataFrame, *, base_bps: float = 10.0,
                      impact_coef: float = 20.0, ref_window: int = 252) -> pd.DataFrame:
    """Date×Code の片道コスト bps パネル＝ base + impact_coef × max(Amihud/median − 1, 0)。

    Amihud/median ≈ 1（平常）→ base。低流動（Amihud 上振れ）でコスト増。≥ base を保証（保守）。
    すべて trailing（≤t・PIT）。engine.backtest(costs_bps=このパネル) に渡す。
    """
    med = amihud.rolling(ref_window, min_periods=20).median()
    norm = (amihud / med.replace(0.0, np.nan))
    excess = (norm - 1.0).clip(lower=0.0).fillna(0.0)        # 平常超過（低流動度）
    return base_bps + impact_coef * excess


def cost_sensitivity_coefs(center: float = 20.0) -> list[float]:
    """感応度スイープ用の impact_coef 候補（保守側中心に前後）。最適化でなく確認のため固定集合。"""
    return [center * 0.5, center, center * 2.0]
