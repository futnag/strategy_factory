"""所有構造オーバーレイ（PEAD スリーブ用・任意・**既定 OFF**）。設計・根拠＝docs/50。

Jinushi(2023) 再現＝PEAD は**高外国人銘柄で消失/反転**（306名で −0.46・ユニバースサイズ
160/263/306 で一貫＝頑健）。本モジュールは PEAD シグナルから高外国人銘柄を除外/減量する
**純関数オーバーレイ**。`enabled=False`（既定）は**恒等＝no-op**で本番に一切影響しない。

注意（過大主張しない）: 未認定（scope=pead_ownership_tilt DSR0.42）かつ foreign≈size
（foreign-beyond-size は標本で反転＝独立αではない）。頑健な核は「高外国人＝一貫した負ドリフト
の除去」というリスク管理。フォワードテスト候補であって認定アップグレードではない。本番恒久ONは
所有データの定期/自動フィードが整ってから（docs/50 §実装注記）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OWNERSHIP_PATH = "data/edinet/ownership_categories.parquet"


def load_ownership(path: str = OWNERSHIP_PATH) -> pd.DataFrame:
    """所有パネル → 銘柄別の**持続値（期間 median）** `[Code, foreign_pct, individual_pct]`。

    所有構造は年1回・粘着的なので median を持続タイプとして用いる（先読みなし）。
    ファイルが無ければ空 DataFrame（呼び出し側で no-op 化）。
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["Code", "foreign_pct", "individual_pct"])
    own = pd.read_parquet(p)
    if own.empty or "Code" not in own.columns:
        return pd.DataFrame(columns=["Code", "foreign_pct", "individual_pct"])
    own = own.assign(Code=own["Code"].astype(str))
    cols = [c for c in ("foreign_pct", "individual_pct") if c in own.columns]
    return own.groupby("Code")[cols].median().reset_index()


def high_foreign_codes(own: pd.DataFrame, cutoff_q: float = 0.67) -> set:
    """`foreign_pct` がクロスセクション分位 `cutoff_q` 以上の銘柄集合（＝除外対象の高外国人帯）。"""
    if own is None or own.empty or "foreign_pct" not in own.columns:
        return set()
    fp = own.dropna(subset=["foreign_pct"])
    if fp.empty:
        return set()
    thr = fp["foreign_pct"].quantile(cutoff_q)
    return set(fp.loc[fp["foreign_pct"] >= thr, "Code"].astype(str))


def apply_high_foreign_exclusion(signal: pd.DataFrame, own: pd.DataFrame | None = None, *,
                                 enabled: bool = False, cutoff_q: float = 0.67,
                                 mode: str = "exclude", shrink: float = 0.0) -> pd.DataFrame:
    """PEAD シグナル wide（index=日付, col=Code）に高外国人除外/減量を適用。

    - `enabled=False`（既定）→ **signal をそのまま返す（no-op・本番無影響）**。
    - `mode="exclude"`    → 高外国人列を NaN（L/S ランキングから除外）。
    - `mode="downweight"` → 高外国人列に `shrink`(∈[0,1)) を乗じて減量（ソフト版）。
    - 所有データが無い銘柄は **除外しない**（unknown=保持＝graceful degrade）。
    - `own` 未指定なら `load_ownership()`（無ければ自動 no-op）。
    """
    if not enabled:
        return signal
    if own is None:
        own = load_ownership()
    hi = high_foreign_codes(own, cutoff_q)
    if not hi:
        return signal
    cols = [c for c in signal.columns if str(c) in hi]
    if not cols:
        return signal
    out = signal.copy()
    if mode == "downweight":
        out[cols] = out[cols] * float(shrink)
    else:
        out[cols] = np.nan
    return out
