"""月次価格パネルの組立とリターン計算。

無料枠の負荷を抑えるため、日次株価は「月末営業日のスナップショット」
（/equities/bars/daily を date 指定で全銘柄一括取得）だけを使う。
1か月あたり1回のAPI呼び出しで全銘柄を取得でき、Parquetキャッシュされる。

純関数（assemble_panel / forward_returns / trailing_momentum）はネットワーク
不要でテスト可能。fetch_month_end_snapshots のみがAPIに触れる。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..data.sources import jquants as jq


def _month_ends(start: str, end: str) -> list[pd.Timestamp]:
    """'YYYY-MM' 〜 'YYYY-MM'（両端含む）の各月の暦上の月末日。"""
    months = pd.period_range(start=start, end=end, freq="M")
    return [m.to_timestamp(how="end").normalize() for m in months]


def fetch_month_end_snapshots(start: str, end: str, max_back: int = 7,
                              refresh: bool = False) -> dict[pd.Timestamp, pd.DataFrame]:
    """各月の月末営業日の全銘柄スナップショットを取得。

    暦月末から最大 max_back 日遡って最初の非空（営業日）を採用する。
    返り値: {実営業日(Timestamp): 日次株価DataFrame}
    """
    snaps: dict[pd.Timestamp, pd.DataFrame] = {}
    for cal_end in _month_ends(start, end):
        for i in range(max_back + 1):
            d = (cal_end - pd.Timedelta(days=i))
            try:
                q = jq.fetch_daily_quotes(d.strftime("%Y%m%d"), refresh=refresh)
            except Exception:  # noqa: BLE001  範囲外/一時エラーはスキップ
                continue
            if not q.empty:
                # 実際の取引日（レスポンスの Date）をラベルに採用
                label = pd.Timestamp(q["Date"].iloc[0]).normalize() \
                    if "Date" in q.columns else d
                snaps[label] = q
                break
    return dict(sorted(snaps.items()))


def assemble_panel(snapshots: dict[pd.Timestamp, pd.DataFrame], value_col: str,
                   code_col: str = "Code") -> pd.DataFrame:
    """スナップショット辞書 → wide パネル（index=日付, columns=Code, 値=value_col）。"""
    series = {}
    for dt, df in snapshots.items():
        if value_col in df.columns and code_col in df.columns:
            s = df.set_index(code_col)[value_col]
            s = s[~s.index.duplicated(keep="last")]
            series[dt] = s
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).T.sort_index()
    panel.columns = [str(c) for c in panel.columns]
    return panel


def forward_returns(price: pd.DataFrame) -> pd.DataFrame:
    """t→t+1 の単純リターンを t 時点にラベル付け（調整後価格 AdjC を使う）。

    panel.pct_change().shift(-1): 行 t の値は「t に建て t+1 で実現するリターン」。
    最終行は NaN（将来未実現）。先読みは無い。
    """
    return price.pct_change().shift(-1)


def trailing_momentum(price: pd.DataFrame, lookback: int = 12,
                      skip: int = 1) -> pd.DataFrame:
    """12-1 モメンタム等：直近 skip か月を除く lookback か月の累積リターン。

    price.shift(skip)/price.shift(lookback) - 1 を t にラベル付け。
    t の値は t-1 以前の価格のみ使用＝先読み無し。
    """
    return price.shift(skip) / price.shift(lookback) - 1.0


def load_daily_panel(field: str = "AdjC", codes: Optional[Iterable] = None,
                     start=None, end=None, base: Optional[str] = None,
                     subdir: str = "daily") -> pd.DataFrame:
    """by-date 日次ミラー（daily/）から wide パネル（index=日付, col=Code, 値=field）を組立。

    柱D（ペア/平均回帰）は日次価格パネルを要するため、全件 by-date Parquet を連結→ピボット
    する（load_fundamentals と同じミラー結合パターン）。各日の実値のみ＝先読みなし、ネット
    ワーク不要（キャッシュ参照のみ）。field 既定は分割調整後終値 AdjC。codes/start/end で限定。
    """
    root = Path(base) if base is not None else jq._CACHE
    d = root / subdir
    frames = []
    if d.exists():
        for p in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(p)
            if df.empty or "_empty" in df.columns:           # 祝日/無データ marker
                continue
            if {"Date", "Code", field}.issubset(df.columns):
                frames.append(df[["Date", "Code", field]])
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    long["Date"] = pd.to_datetime(long["Date"]).dt.normalize()
    long["Code"] = long["Code"].astype(str)
    if codes is not None:
        want = {str(c) for c in codes}
        long = long[long["Code"].isin(want)]
    long = long.drop_duplicates(subset=["Date", "Code"], keep="last")
    if long.empty:
        return pd.DataFrame()
    panel = long.pivot(index="Date", columns="Code", values=field).sort_index()
    panel.columns = [str(c) for c in panel.columns]
    if start is not None:
        panel = panel.loc[pd.Timestamp(start):]
    if end is not None:
        panel = panel.loc[:pd.Timestamp(end)]
    return panel
