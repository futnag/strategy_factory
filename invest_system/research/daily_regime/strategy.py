"""daily_regime: 一次戦略＋レジーム適用（フィルタ／連続サイジング）— step 4。

§6.2 の3経路のうち**フィルタ**と**動的サイジング**を、§2 の PEAD 適用除外つきで実装。
レジーム値（prob_stressed）・決算窓フラグは AsOf フレームから ≤t で読む（PIT 強制）。

§2 継ぎ目（接続設計 6.3・3C）：event_driven かつ**決算窓内**では
  - フィルタ（新規ポジション抑制）を**適用しない**（決算後ドリフトを殺さない）。
  - ただし**サイジングは残す**（流動性チャネル由来のレジーム値で縮小）。
それ以外（非イベント戦略・窓外）は高ストレスでフィルタ＋サイジング（通常規則）。

一次戦略は §6.1 のとおり**粗く固定**（最適化しない）。

契約：**フィールド欠落（配線ミス）は例外**・値の欠損（当日情報なし）は優雅な劣化。
silent fallback（欠落フィールド→空/0.0）はデータバグを「エッジなし」に化けさせるため禁止。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from invest_system.research.strategy import Strategy


class ConstantLong(Strategy):
    """常時ロング（B&H・no-filter の素地）。"""

    def __init__(self, code: str, side: float = 1.0):
        self.code = str(code)
        self.side = float(side)
        self.name = f"const_long({code})"
        self.params = {"code": code}

    def target_weights(self, asof) -> pd.Series:
        cl = asof.frame("close")
        if self.code not in cl.columns or not len(cl) or pd.isna(cl[self.code].iloc[-1]):
            return pd.Series(dtype="float64")
        return pd.Series({self.code: self.side}, dtype="float64")


class MomentumPrimary(Strategy):
    """粗いトレンド一次戦略：trailing lookback リターン>0 でロング・否なら現金（固定・非最適化）。"""

    def __init__(self, code: str, lookback: int = 60, side: float = 1.0):
        self.code = str(code)
        self.lookback = int(lookback)
        self.side = float(side)
        self.name = f"mom_primary({code},lb={lookback})"
        self.params = {"code": code, "lookback": lookback}

    def target_weights(self, asof) -> pd.Series:
        cl = asof.frame("close")
        if self.code not in cl.columns:
            return pd.Series(dtype="float64")
        s = cl[self.code].dropna()
        if len(s) < self.lookback + 1:
            return pd.Series(dtype="float64")
        if s.iloc[-1] / s.iloc[-1 - self.lookback] - 1.0 > 0.0:
            return pd.Series({self.code: self.side}, dtype="float64")
        return pd.Series(dtype="float64")


class PostEarningsDrift(Strategy):
    """粗い PEAD：決算窓フラグ（earn_field）が True の間ロング（決算後ドリフトを取る）。"""

    def __init__(self, code: str, earn_field: str = "earn_window", side: float = 1.0):
        self.code = str(code)
        self.earn_field = earn_field
        self.side = float(side)
        self.name = f"pead({code})"
        self.params = {"code": code}

    def target_weights(self, asof) -> pd.Series:
        f = asof.frame(self.earn_field)      # フィールド欠落＝配線ミス → 例外
        earn = f.iloc[-1] if len(f) else None
        if earn is None or self.code not in earn.index or not bool(earn.get(self.code, False)):
            return pd.Series(dtype="float64")
        return pd.Series({self.code: self.side}, dtype="float64")


class ValueReversalPrimary(Strategy):
    """E/P =（DiscDate アンカー EPS_asof）/close、自己 trailing 中央値超でロング。

    ⚠ 注記：EPS_asof は開示時のみ更新の階段関数ゆえ、日次 E/P 変動の主因は分母 close。よって本シグナルは
    ファンダ割安でなく**価格の自己相対水準での逆張り（短期リバーサル）寄り**（解釈に注意）。固定・非最適化。
    """

    def __init__(self, code: str, lookback: int = 756, eps_field: str = "eps_asof"):
        self.code = str(code)
        self.lookback = int(lookback)
        self.eps_field = eps_field
        self.name = f"value_rev({code},lb={lookback})"
        self.params = {"code": code, "lookback": lookback}

    def target_weights(self, asof) -> pd.Series:
        cl = asof.frame("close")
        ef = asof.frame(self.eps_field)      # フィールド欠落＝配線ミス → 例外
        if self.code not in cl.columns or self.code not in ef.columns:
            return pd.Series(dtype="float64")
        ep = (ef[self.code] / cl[self.code]).dropna()
        if len(ep) < max(20, self.lookback // 4):
            return pd.Series(dtype="float64")
        cur, med = ep.iloc[-1], ep.iloc[-self.lookback:].median()
        if pd.notna(cur) and pd.notna(med) and cur >= med:
            return pd.Series({self.code: 1.0}, dtype="float64")
        return pd.Series(dtype="float64")


class PEADSurprisePrimary(Strategy):
    """正 FY サプライズ後の決算窓（pead_window True）でロング（genuine PEAD ドリフト）。"""

    def __init__(self, code: str, pead_field: str = "pead_window"):
        self.code = str(code)
        self.pead_field = pead_field
        self.name = f"pead_surprise({code})"
        self.params = {"code": code}

    def target_weights(self, asof) -> pd.Series:
        f = asof.frame(self.pead_field)      # フィールド欠落＝配線ミス → 例外
        w = f.iloc[-1] if len(f) else None
        if w is None or self.code not in w.index or not bool(w.get(self.code, False)):
            return pd.Series(dtype="float64")
        return pd.Series({self.code: 1.0}, dtype="float64")


class RegimeAwareStrategy(Strategy):
    """一次戦略に レジームフィルタ＋連続サイジング を適用（§2 PEAD 適用除外つき）。

    event_driven=True かつ決算窓内では**フィルタ非適用（新規を止めない）・サイジングは残す**。
    それ以外は high-stress でフィルタ（新規抑制）＋サイジング。size = max(size_floor, 1 − prob_stressed)。
    """

    def __init__(self, primary: Strategy, *, regime_field: str = "prob_stressed",
                 earn_field: str = "earn_window", filter_thresh: float = 0.6,
                 size_floor: float = 0.2, event_driven: bool = False,
                 name: Optional[str] = None):
        self.primary = primary
        self.regime_field = regime_field
        self.earn_field = earn_field
        self.filter_thresh = float(filter_thresh)
        self.size_floor = float(size_floor)
        self.event_driven = bool(event_driven)
        self.name = name or f"{primary.name}|regime{'(pead-exempt)' if event_driven else ''}"
        self.params = {**getattr(primary, "params", {}), "filter_thresh": filter_thresh,
                       "size_floor": size_floor, "event_driven": event_driven}

    def _stressed(self, asof, code: str) -> float:
        f = asof.frame(self.regime_field)    # フィールド欠落＝配線ミス → 例外
        row = f.iloc[-1] if len(f) else None
        if row is None or code not in row.index or pd.isna(row.get(code)):
            return 0.0                        # 値の欠損（当日情報なし）＝フィルタ無効の優雅な劣化
        return float(row.get(code))

    def _in_earn(self, asof, code: str) -> bool:
        if not self.event_driven:
            return False
        f = asof.frame(self.earn_field)      # event_driven に決算窓は必須 → 欠落は例外
        row = f.iloc[-1] if len(f) else None
        return bool(row.get(code, False)) if (row is not None and code in row.index) else False

    def target_weights(self, asof) -> pd.Series:
        w = self.primary.target_weights(asof)
        if w.empty:
            return w
        out = {}
        for code in w.index:
            s = self._stressed(asof, code)
            in_earn = self._in_earn(asof, code)
            size = max(self.size_floor, 1.0 - s)             # 連続サイジング（流動性枯渇で縮小）
            if in_earn:
                out[code] = float(w[code]) * size            # §2：フィルタ除外・サイジング残す
            elif s >= self.filter_thresh:
                out[code] = 0.0                               # 通常フィルタ（新規抑制）
            else:
                out[code] = float(w[code]) * size
        res = pd.Series(out, dtype="float64")
        return res[res != 0.0]
