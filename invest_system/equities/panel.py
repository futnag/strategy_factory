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

    fill_method=None を明示：欠損を前方補完せず NaN のまま比率を取る（pandas 既定の
    変更に依らず挙動を固定。例：上場前/欠測月のリターンを偽の 0% にしない）。
    """
    return price.pct_change(fill_method=None).shift(-1)


def delisting_mask(price: pd.DataFrame) -> pd.DataFrame:
    """各銘柄の最終取引月（最後の非NaN）がパネル終端より前なら「廃止」＝その月を True。

    前月在籍→以降消滅で上場廃止を検知（J-Quants に廃止フラグが無いため消滅から推定）。
    パネル終端まで在籍する銘柄は廃止でない（データ末尾＝現存）。散発的な欠測月は最終
    非NaN が後ろにずれるため誤検知しない。
    """
    mask = pd.DataFrame(False, index=price.index, columns=price.columns)
    if price.empty:
        return mask
    last_end = price.index[-1]
    for code in price.columns:
        valid = price[code].dropna()
        if valid.empty:
            continue
        m_last = valid.index[-1]
        if m_last < last_end:
            mask.loc[m_last, code] = True
    return mask


def impute_delisting(returns: pd.DataFrame, price: pd.DataFrame,
                     policy: str = "last_price", delist_value: float = -1.0,
                     heuristic_threshold: float = -0.5) -> pd.DataFrame:
    """廃止月の（欠損する）フォワードリターンを方針で補完＝生存者バイアスを正す。

    廃止月は price[t+1] が NaN のため forward_returns が NaN になり、その銘柄の最終損益が
    クロスセクションから落ちて上方バイアスになる。J-Quants は廃止理由/最終気配を持たない
    ため、理由別ターミナル値は導けない。方針（config 可変）：
      - last_price（既定）：最終気配で清算＝増分 0（偽の損失を作らない）。
      - all_minus100：全廃止を全損 ＝ delist_value（既定 −1.0）。診断のバイアス下限。
      - heuristic：直近月次（→m_last）が heuristic_threshold 未満なら全損（倒産プロキシ）、
        他は 0（TOB/市場変更プロキシ）。
    """
    out = returns.copy()
    mask = delisting_mask(price)
    if not bool(mask.to_numpy().any()):
        return out
    for code in price.columns:
        col = mask[code]
        if not col.any():
            continue
        m_last = col[col].index[0]
        if policy == "all_minus100":
            imp = delist_value
        elif policy == "heuristic":
            r = price[code].pct_change(fill_method=None).loc[m_last]
            imp = delist_value if (pd.notna(r) and r < heuristic_threshold) else 0.0
        else:                                            # last_price（既定）
            imp = 0.0
        out.loc[m_last, code] = imp
    return out


def forward_returns_with_delisting(price: pd.DataFrame, policy: str = "last_price",
                                   delist_value: float = -1.0,
                                   heuristic_threshold: float = -0.5) -> pd.DataFrame:
    """forward_returns に廃止リターン補完を重ねた便宜関数（価格リターンのみ）。"""
    return impute_delisting(forward_returns(price), price, policy=policy,
                            delist_value=delist_value,
                            heuristic_threshold=heuristic_threshold)


def dividend_per_share_asof(rebal_dates, codes: Optional[Iterable] = None,
                            lag_days: int = 1, base: Optional[str] = None,
                            field: str = "DivAnn") -> pd.DataFrame:
    """実績 年間配当 DPS を as-of（DiscDate≤t−lag）で wide 化（PIT）。fins_summary 由来。

    point_in_time が文字列の DivAnn を数値化して as-of 突合する。返り値 index=rebal, col=Code。
    """
    from .fundamentals import load_fundamentals, point_in_time
    rebal = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebal_dates)))).normalize()
    fund = load_fundamentals(codes=codes, base=base)
    if fund.empty or field not in fund.columns:
        return pd.DataFrame(index=rebal, dtype="float64")
    return point_in_time(fund, rebal, [field], lag_days=lag_days)[field]


def total_forward_returns(adj_price: pd.DataFrame, raw_price: Optional[pd.DataFrame] = None,
                          dps: Optional[pd.DataFrame] = None,
                          periods_per_year: int = 12) -> pd.DataFrame:
    """配当込みトータル・フォワードリターン＝価格（調整後）フォワード ＋ 配当利回り寄与。

    adj_price：調整後終値 AdjC（分割調整済＝価格リターンの素）。raw_price：生終値 C
    （配当利回りの分母＝DPS と分割基準を揃える）。dps：実績年間 DPS の as-of wide。
    配当寄与は (DPS / periods_per_year) / raw_price を各月に均等計上（PIT：DPS は as-of、
    価格は当月）。raw_price/dps が無ければ価格リターンのみ＝forward_returns と一致（後方互換）。
    """
    fwd = forward_returns(adj_price)
    if raw_price is None or dps is None or raw_price.empty or dps.empty:
        return fwd
    denom = raw_price.reindex(index=fwd.index, columns=fwd.columns).where(lambda x: x > 0)
    d = dps.reindex(index=fwd.index, columns=fwd.columns)
    div_yield = (d / float(periods_per_year)) / denom
    return fwd + div_yield.fillna(0.0)


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

    既定（base=None）では **Silver 層（processed/equities/wide）を優先**して O(1) で読む
    （`store.materialize_*` 生成済みなら高速）。未生成なら Raw by-date を連結→ピボットへ
    フォールバック（後方互換）。各日の実値のみ＝先読みなし・ネット不要。field 既定は調整後
    終値 AdjC（"AdjC"/"C"/"Va" 等の別名は Silver 側で解決）。codes/start/end で限定。
    """
    if base is None:                                  # Silver 高速パス（既定 data root）
        from ..data.store import load_wide
        wide = load_wide(field, start=start, end=end, base=str(jq._CACHE.parent))
        if not wide.empty:
            if codes is not None:
                want = {str(c) for c in codes}
                wide = wide.reindex(columns=[c for c in wide.columns if c in want])
            return wide
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
