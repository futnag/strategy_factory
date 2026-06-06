"""PITデータアクセス：戦略に「基準日以前」しか見せない構造（リーク遮断の核）。

戦略は AsOfView.asof(t) が返す AsOf からのみデータを読む。AsOf は構築時に
「日付 ≤ t」へスライス済みなので、未来のバーを参照する術が無い（=構造的に先読み
不能）。戦略作者の善意に依存しない、フレームワーク強制のリーク防止。

パネルは wide（index=取引日 tz-naive, columns=銘柄コード）。最低 close を要求し、
open/high/low/volume があれば併せて公開する。
"""
from __future__ import annotations

import pandas as pd


class AsOf:
    """基準日 asof の時点で参照可能なデータ（未来は含まない）。"""

    def __init__(self, panels: dict[str, pd.DataFrame], asof: pd.Timestamp):
        self._p = panels
        self.asof = asof

    def frame(self, field: str) -> pd.DataFrame:
        """field の wide パネル（asof までの全行）。"""
        if field not in self._p:
            raise KeyError(f"field '{field}' は未提供（利用可能: {list(self._p)}）")
        return self._p[field]

    def last(self, field: str) -> pd.Series:
        """最新行（= asof 当日 or 直近営業日）の銘柄横断スナップショット。"""
        f = self._p[field]
        return f.iloc[-1] if len(f) else pd.Series(dtype="float64")

    def lag(self, field: str, k: int = 1) -> pd.Series:
        """k 本前の行（k=1 で前日）。不足時は空 Series。"""
        f = self._p[field]
        return f.iloc[-1 - k] if len(f) > k else pd.Series(dtype="float64")

    def n_bars(self) -> int:
        return len(self._p["close"])


class AsOfView:
    """全期間パネルを保持し、asof(t) で「t以前」だけ見える AsOf を発行する。"""

    def __init__(self, panels: dict[str, pd.DataFrame]):
        if "close" not in panels:
            raise ValueError("panels には少なくとも 'close' が必要です。")
        # tz-naive に正規化（パイプライン規約）。列順は close に揃える。
        self.panels = {k: v.copy() for k, v in panels.items()}
        self.close = self.panels["close"]
        self.dates = self.close.index
        self.codes = list(self.close.columns)

    def asof(self, date) -> AsOf:
        d = pd.Timestamp(date)
        sliced = {k: v.loc[:d] for k, v in self.panels.items()}  # ≤ d のみ
        return AsOf(sliced, d)
