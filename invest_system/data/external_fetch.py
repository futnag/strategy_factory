"""外部価格（data/investers/）の無人差分更新（Phase 2 運用・D5）。

正本の investers/ は investing.com の手動エクスポート由来。無人運用では
この手動依存が単一障害点になるため、本モジュールが**既存履歴は不変のまま**
（研究データ凍結＝DP10）、新規日付のみ Yahoo Finance から追記する。

安全装置（採用前に必ず通る）：
- 重複日クロスチェック … 直近の共通日における中央値|乖離|が tolerance
  （既定2%）超のソースは**不採用**（別系列・単位違い・通貨違いの混入を遮断）。
  シンボル表は 2026-06 に実履歴と重複日照合済み（最大でも銅の 64bp）。
- 追記は old.index.max() より**後**の日付のみ＝過去の書換えをしない（冪等）。
- **確定足のみ追記**（cutoff）… 対象シンボル（CME先物・FX）はほぼ24時間取引で、
  取得時点（21:30 JST 等）の「当日」日足は形成途中。過去不変の追記設計では
  部分足を一度取り込むと永久凍結されるため、UTC 当日以降の行は追記しない
  （確定した足だけが翌日以降に追記される＝鮮度を1日譲って正しさを取る）。

ネットワーク層（fetch_yahoo・yfinance 遅延 import）と純関数
（validate_overlap / merge_new_dates）を分離し、純関数はオフラインでテストする。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .external import _PRICE_FILES

# Yahoo Finance シンボル（2026-06 に investers/ 実履歴と重複日照合済み）
YAHOO_SYMBOLS: dict[str, str] = {
    "nk225_fut": "NIY=F",       # CME 日経225先物（円建）。OSE 主限月と ~30bp
    "sp500": "^GSPC",
    "nasdaq_comp": "^IXIC",
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "copper": "HG=F",
    "wti": "CL=F",
    "usdjpy": "USDJPY=X",
    "eurjpy": "EURJPY=X",
    "audjpy": "AUDJPY=X",
}
# Phase 2 が必要とする全キー（TSMOM 11資産。nk225_fut はヘッジ価格も兼ねる）
PHASE2_KEYS = list(YAHOO_SYMBOLS)
_COLS = ["open", "high", "low", "close", "volume", "change_pct"]


def fetch_yahoo(symbol: str, period: str = "3mo") -> pd.DataFrame:
    """Yahoo から日次 OHLCV を investers/ スキーマ（小文字・naive index）で返す。"""
    import yfinance as yf  # 遅延 import（コア依存にしない）

    df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.DataFrame({
        "open": pd.to_numeric(df["Open"], errors="coerce"),
        "high": pd.to_numeric(df["High"], errors="coerce"),
        "low": pd.to_numeric(df["Low"], errors="coerce"),
        "close": pd.to_numeric(df["Close"], errors="coerce"),
        "volume": pd.to_numeric(df.get("Volume"), errors="coerce"),
    })
    out.index = idx.normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["close"])


def validate_overlap(old_close: pd.Series, new_close: pd.Series, *,
                     min_overlap: int = 5, lookback: int = 60,
                     tolerance: float = 0.02) -> tuple[bool, float, int]:
    """直近 lookback 日の共通日で中央値|乖離|を測る → (ok, med_diff, n)。

    n < min_overlap は照合不能＝不採用（False）。old が空でも False
    （無検証の系列差し替えを許さない）。
    """
    o, n_ = old_close.dropna(), new_close.dropna()
    common = o.index.intersection(n_.index).sort_values()[-lookback:]
    if len(common) < min_overlap:
        return False, float("nan"), int(len(common))
    diff = float((n_.loc[common] / o.loc[common] - 1.0).abs().median())
    return bool(diff <= tolerance), diff, int(len(common))


def _utc_today() -> pd.Timestamp:
    """UTC の「今日」（naive 00:00）。形成中の日足を遮断する既定カットオフ。"""
    return pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)


def merge_new_dates(old: pd.DataFrame, new: pd.DataFrame, *,
                    cutoff: pd.Timestamp | None = None
                    ) -> tuple[pd.DataFrame, int]:
    """old の最終日より**後**かつ cutoff より**前**の行だけ new から追記（過去は不変・冪等）。

    cutoff: この日付**以降**の行は追記しない（モジュール docstring「確定足のみ追記」）。
    None は無条件（後方互換）。update_external_prices は既定で UTC 当日を渡す。
    change_pct は境界を跨いだ close 前日比（%）で新規行のみ再計算する。
    """
    if new.empty:
        return old, 0
    last = old.index.max() if len(old) else pd.Timestamp.min
    add = new[new.index > last]
    if cutoff is not None:
        add = add[add.index < pd.Timestamp(cutoff)]
    if add.empty:
        return old, 0
    add = add.reindex(columns=["open", "high", "low", "close", "volume"]).copy()
    prev = (pd.concat([old["close"].iloc[-1:], add["close"]])
            if len(old) else add["close"])
    add["change_pct"] = (prev.pct_change() * 100).reindex(add.index).round(2)
    merged = pd.concat([old.reindex(columns=_COLS), add]).sort_index()
    merged.index.name = old.index.name or "date"
    return merged, int(len(add))


def update_external_prices(keys=None, *, base: str = "data", period: str = "3mo",
                           tolerance: float = 0.02, fetch=None,
                           cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """keys（既定 PHASE2_KEYS）を差分追記し、キー別レポートを返す。

    fetch は差し替え可能（テストはネット不要）。あるキーの失敗は残りを止めない
    （status 列に記録し、鮮度は reconcile の status.json で監視する）。
    cutoff: 既定 None = UTC 当日（形成中の日足を追記しない＝確定足のみ）。
    """
    fetch = fetch or fetch_yahoo
    cutoff = _utc_today() if cutoff is None else pd.Timestamp(cutoff)
    pdir = Path(base) / "investers"
    rows = []
    for k in (PHASE2_KEYS if keys is None else list(keys)):
        fn, sym = _PRICE_FILES.get(k), YAHOO_SYMBOLS.get(k)
        row = {"key": k, "symbol": sym, "status": "", "n_new": 0,
               "last": None, "overlap_bp": np.nan}
        if fn is None or sym is None:
            row["status"] = "NO-MAP"
            rows.append(row)
            continue
        fp = pdir / fn
        try:
            old = (pd.read_parquet(fp) if fp.exists()
                   else pd.DataFrame(columns=_COLS))
            if len(old):
                old.index = pd.to_datetime(old.index).normalize()
            new = fetch(sym, period=period)
            if new.empty:
                row["status"] = "EMPTY"
                row["last"] = old.index.max().date() if len(old) else None
                rows.append(row)
                continue
            ok, diff, n = validate_overlap(old.get("close", pd.Series(dtype=float)),
                                           new["close"], tolerance=tolerance)
            row["overlap_bp"] = diff * 1e4
            if not ok:
                row["status"] = f"REJECT(n={n})"
                row["last"] = old.index.max().date() if len(old) else None
                rows.append(row)
                continue
            merged, n_new = merge_new_dates(old, new, cutoff=cutoff)
            if n_new:
                pdir.mkdir(parents=True, exist_ok=True)
                merged.to_parquet(fp)
            row.update(status="OK", n_new=n_new, last=merged.index.max().date())
        except Exception as e:  # noqa: BLE001 — 1キーの失敗で全体を止めない
            row["status"] = f"FAIL:{str(e)[:48]}"
        rows.append(row)
    return pd.DataFrame(rows)
