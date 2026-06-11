"""Processed (Silver) 層：Raw by-date を分析最適化の wide パネルに materialize。

設計方針：
- Raw (`data/jquants/daily`, by-date) は不変・append-only（取得最適・冪等・再開可能）。
- Processed は **フィールド別 wide**（index=Date, col=Code, 1ファイル/フィールド）＝
  クロスセクション＋ベクトル化の双方に最適、`AsOfView` が直接消費できる。
- **生(無調整)は純 append／調整後(adj_*)は派生**：分割で過去が back-adjust され書き換わる
  ため、`adj_close` 等は「生 × 累積調整係数」で再構築する（append では不可）。
- 補助：partitioned-long（year パーティション）＝DuckDB/SQL・将来のティック/分足スケール用。

pandas/pyarrow のみ・ネット不要。base 既定="data"（データroot）。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Silver 正準フィールド → Raw 列
_RAW_COL = {
    "open": "O", "high": "H", "low": "L", "close": "C",
    "volume": "Vo", "turnover": "Va", "adj_factor": "AdjFactor",
    "upper_limit": "UL", "lower_limit": "LL",
}
# 既存呼び出し（"AdjC"/"C"/"Va"...）→ 正準フィールド別名（後方互換）
_FIELD_ALIAS = {
    "O": "open", "H": "high", "L": "low", "C": "close", "Vo": "volume",
    "Va": "turnover", "AdjFactor": "adj_factor", "UL": "upper_limit",
    "LL": "lower_limit", "AdjC": "adj_close", "AdjO": "adj_open",
    "AdjH": "adj_high", "AdjL": "adj_low",
}
_KEEP = ["Date", "Code", "O", "H", "L", "C", "Vo", "Va", "AdjFactor", "UL", "LL"]


def _wide_dir(base) -> Path:
    return Path(base) / "processed" / "equities" / "wide"


def _raw_dir(base) -> Path:
    return Path(base) / "jquants" / "daily"


def _read_raw_long(raw_dir: Path, skip_dates: Optional[set] = None) -> pd.DataFrame:
    """by-date Raw を必要列だけ連結（空マーカー/取得済み日はスキップ）。"""
    frames = []
    for p in sorted(raw_dir.glob("*.parquet")):
        if skip_dates is not None and p.stem in skip_dates:
            continue
        df = pd.read_parquet(p)
        if df.empty or "_empty" in df.columns:
            continue
        frames.append(df[[c for c in _KEEP if c in df.columns]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def materialize_wide(fields: Optional[Iterable[str]] = None, base: str = "data",
                     incremental: bool = True) -> dict:
    """Raw by-date → フィールド別 wide（`processed/equities/wide/<field>.parquet`）。

    incremental=True は「**対象フィールド全部に**収録済みの日付だけ」をスキップして
    Raw から読み append（冪等）。close 単独基準だと、後から追加したフィールドが
    「close に収録済みの日付」を全部飛ばされて空のままになるため、フィールド別に
    判定する（1つでも未収録ならその日付は再読込＝既収録フィールド側は同値上書きで
    冪等）。新規上場は列追加・退場は以降 NaN（生存者バイアス無しを自然保持）。
    """
    out_dir = _wide_dir(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(_RAW_COL) if fields is None else list(fields)
    skip = None
    if incremental:
        mapped = [f for f in fields if f in _RAW_COL]
        fps = [out_dir / f"{f}.parquet" for f in mapped]
        if fps and all(fp.exists() for fp in fps):
            sets = []
            for fp in fps:                       # columns=[] で index だけ読む（軽量）
                idx = pd.to_datetime(pd.read_parquet(fp, columns=[]).index)
                sets.append({d.strftime("%Y%m%d") for d in idx})
            skip = set.intersection(*sets)
    long = _read_raw_long(_raw_dir(base), skip_dates=skip)
    if long.empty:
        return {"appended_dates": 0, "fields": 0}
    long["Date"] = pd.to_datetime(long["Date"]).dt.normalize().astype("datetime64[ns]")
    long["Code"] = long["Code"].astype(str)
    long = long.drop_duplicates(["Date", "Code"], keep="last")
    n_new = int(long["Date"].nunique())
    done = 0
    for f in fields:
        col = _RAW_COL.get(f)
        if col is None or col not in long.columns:
            continue
        new_wide = long.pivot(index="Date", columns="Code", values=col)
        new_wide.columns = [str(c) for c in new_wide.columns]
        fp = out_dir / f"{f}.parquet"
        if fp.exists():
            comb = pd.concat([pd.read_parquet(fp), new_wide])
            comb = comb[~comb.index.duplicated(keep="last")].sort_index()
        else:
            comb = new_wide.sort_index()
        comb.to_parquet(fp)
        done += 1
    return {"appended_dates": n_new, "fields": done}


def rebuild_adjusted(base: str = "data",
                     price_fields=("close", "open", "high", "low")) -> dict:
    """生 OHLC ＋ 日次 adj_factor から back-adjusted（adj_*）を再構築。

    adj[t] = price[t] × Π(adj_factor[s] for s>t)。最新日は係数1（無調整）。分割は過去を
    書き換えるため append でなく全体再計算（新しい adj_factor≠1 が現れたら呼ぶ）。
    """
    wide = _wide_dir(base)
    af_fp = wide / "adj_factor.parquet"
    if not af_fp.exists():
        return {"error": "adj_factor not materialized"}
    af = pd.read_parquet(af_fp).sort_index().astype("float64")
    af = af.where(af.notna(), 1.0)
    cum_incl = af[::-1].cumprod()[::-1]          # Π_{s>=t}
    cum_future = cum_incl.shift(-1)              # Π_{s>t}
    cum_future.iloc[-1] = 1.0
    cum_future = cum_future.where(cum_future.notna(), 1.0)
    out = {}
    for f in price_fields:
        fp = wide / f"{f}.parquet"
        if not fp.exists():
            continue
        raw = pd.read_parquet(fp).sort_index()
        cf = cum_future.reindex(index=raw.index, columns=raw.columns)
        cf = cf.where(cf.notna(), 1.0)
        (raw * cf).to_parquet(wide / f"adj_{f}.parquet")
        out[f"adj_{f}"] = [int(raw.shape[0]), int(raw.shape[1])]
    return out


def load_wide(field: str, start=None, end=None, base: str = "data") -> pd.DataFrame:
    """Silver wide パネルを読む（高速・O(1)）。'AdjC'/'C'/'Va' 等の別名も解決。"""
    canon = _FIELD_ALIAS.get(field, field)
    fp = _wide_dir(base) / f"{canon}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index)
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]
    return df


def materialize_long(base: str = "data") -> dict:
    """補助：partitioned-long（year パーティション・DuckDB/SQL用）を Raw から全再構築。

    日次更新では不要（wide のみ更新）。週次/オンデマンドで実行する想定。
    """
    out = Path(base) / "processed" / "equities" / "long"
    long = _read_raw_long(_raw_dir(base))
    if long.empty:
        return {"rows": 0}
    long["Date"] = pd.to_datetime(long["Date"]).dt.normalize()
    long["Code"] = long["Code"].astype(str)
    long = long.drop_duplicates(["Date", "Code"], keep="last")
    long["year"] = long["Date"].dt.year
    long = long.sort_values(["year", "Code", "Date"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, partition_cols=["year"])
    return {"rows": int(len(long)), "years": int(long["year"].nunique())}


def materialize_all(base: str = "data", incremental: bool = True,
                    long: bool = False) -> dict:
    """日次運用の標準手順：wide 増分 → 調整再構築（係数変化時のみ実質変化）。"""
    rep = {"wide": materialize_wide(base=base, incremental=incremental),
           "adjusted": rebuild_adjusted(base=base)}
    if long:
        rep["long"] = materialize_long(base=base)
    return rep


def health_check(base: str = "data") -> pd.DataFrame:
    """各 wide フィールドの被覆・NaN率・期間と、フィールド間の日付整合を点検。"""
    wide = _wide_dir(base)
    rows, indices = [], {}
    for fp in sorted(wide.glob("*.parquet")):
        df = pd.read_parquet(fp)
        idx = pd.to_datetime(df.index)
        indices[fp.stem] = idx
        nan_ratio = float(df.isna().to_numpy().mean()) if df.size else float("nan")
        rows.append({"field": fp.stem, "days": len(df), "codes": int(df.shape[1]),
                     "start": idx.min().date() if len(idx) else None,
                     "end": idx.max().date() if len(idx) else None,
                     "nan_pct": round(nan_ratio * 100, 1)})
    ref = indices.get("close")
    for r in rows:
        r["aligned_to_close"] = bool(ref is not None
                                     and indices[r["field"]].equals(ref))
    return pd.DataFrame(rows)
