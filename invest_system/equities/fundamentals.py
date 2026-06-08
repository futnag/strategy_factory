"""財務サマリーのポイントインタイム整合（先読みバイアスの排除）。

各リバランス日 t に対し、各銘柄で「開示日 DiscDate ≤ t − lag_days」を満たす
最新の開示値のみを採用する。決算は場中(14:00等)に開示され得るため lag_days≥1 を
既定とし、開示当日の価格には反映させない保守的設計とする。

これは因果推論（pillar C）以前の最重要規律：未来情報の混入を断つ。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..data.sources import jquants as jq


def _concat_parquets(d: Path) -> pd.DataFrame:
    """dir 内の全Parquetを長形式で連結（空マーカー _empty はスキップ）。"""
    frames = []
    if d.exists():
        for p in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(p)
            if df.empty or "_empty" in df.columns:
                continue
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_fundamentals(codes: Optional[Iterable] = None, base: Optional[str] = None,
                      subdir: str = "fins_summary",
                      also_subdir: Optional[str] = "statements") -> pd.DataFrame:
    """財務サマリーを長形式（行=開示）で読み込む。全件 by-date ミラー fins_summary/ を主とし、
    旧 by-code キャッシュ statements/ も併合・重複除去する（DL途中でも欠落しない）。

    行に DiscDate, Code と各財務フィールド。空マーカーは除外。codes 指定で銘柄限定。
    返り値は point_in_time にそのまま渡せる。全件DL後は fins_summary/ が完全な上位集合となり、
    statements/ 由来は重複として落ちる。同一開示の重複は (Code, DiscDate, DiscNo) で除去。
    """
    root = Path(base) if base is not None else jq._CACHE
    parts = [p for p in (_concat_parquets(root / s)
                         for s in (subdir, also_subdir) if s) if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    keys = [k for k in ("Code", "DiscDate", "DiscNo") if k in df.columns]
    if keys:
        df = df.sort_values(keys).drop_duplicates(subset=keys, keep="last")
    if codes is not None and "Code" in df.columns:
        want = {str(c) for c in codes}
        df = df[df["Code"].astype(str).isin(want)]
    return df.reset_index(drop=True)


def point_in_time(fund_long: pd.DataFrame, rebal_dates, fields: list[str],
                  date_col: str = "DiscDate", code_col: str = "Code",
                  lag_days: int = 1) -> dict[str, pd.DataFrame]:
    """銘柄×開示の長形式 → フィールド別の as-of wide パネル。

    fund_long  : 行=開示, 列に date_col, code_col, 各 field を含む
    rebal_dates: リバランス日の列（Timestamp 群）
    返り値      : {field: DataFrame(index=rebal_dates, columns=code)}（as-of値）
    """
    rebal = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebal_dates)))).normalize()
    present = [f for f in fields if f in fund_long.columns]
    if fund_long.empty or not present:
        return {f: pd.DataFrame(index=rebal, dtype="float64") for f in present}

    df = fund_long.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    # as-of 突合のための左キー（リバランス日からラグを引いた締切日）
    left = pd.DataFrame({"asof": rebal})
    left["cutoff"] = left["asof"] - pd.Timedelta(days=lag_days)
    left = left.sort_values("cutoff")

    # 列を1本ずつ挿入するとDataFrameが断片化する（pandas警告＋低速）。
    # フィールド別に {code: series} を貯め、最後に一括構築する。
    acc: dict[str, dict[str, pd.Series]] = {f: {} for f in present}
    for code, g in df.groupby(code_col):
        g = g.sort_values(date_col)
        # 同一開示日が複数なら最後（訂正等）を採用
        g = g[~g[date_col].duplicated(keep="last")]
        m = pd.merge_asof(left, g, left_on="cutoff", right_on=date_col,
                          direction="backward").set_index("asof")
        for f in present:
            if f in m.columns:
                acc[f][str(code)] = pd.to_numeric(m[f], errors="coerce").reindex(rebal)
    return {f: (pd.DataFrame(acc[f], index=rebal) if acc[f]
               else pd.DataFrame(index=rebal, dtype="float64")) for f in present}


def fundamentals_panel(rebal_dates, fields: list[str], codes: Optional[Iterable] = None,
                       lag_days: int = 1, base: Optional[str] = None
                       ) -> dict[str, pd.DataFrame]:
    """全件 by-date ミラーから財務 as-of パネルを1呼び出しで組み立てる（案A推奨経路）。

    load_fundamentals（fins_summary/ ＋ 旧 statements/ を併合・重複除去）→ point_in_time
    （DiscDate≤t−lag の最新開示のみ採用）を合成。codes=None なら全ユニバース。返り値は
    {field: DataFrame(index=rebal, columns=code)}。銘柄ごとにネットワークを叩かないため、
    全銘柄パネルでも高速・先読みなし。
    """
    return point_in_time(load_fundamentals(codes=codes, base=base), rebal_dates,
                         fields, lag_days=lag_days)
