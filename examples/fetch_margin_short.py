"""信用取引・空売り系データの一括ダウンロード（Standard・10年窓・再開可能）。

効率的な取得軸（API呼び出し最小化）:
  short_ratio  : 業種(S33)別に from/to 全期間 → 約33回
  weekly       : 毎週金曜=全銘柄（祝日は数日遡る）→ 約520回
  positions    : 各営業日 calc_date=全銘柄 → 約2,600回
  alert        : 各営業日 date=規制銘柄 → 約2,600回
全てローカルParquetにキャッシュ（data/jquants/...、gitignore済）。中断しても
キャッシュ済みはスキップして再開可能。Standard(120/分)前提＝J_QUANTS_MIN_INTERVAL=0.7。

データセット選択: 環境変数 J_MS_DATASETS（既定 "ratio,weekly,positions,alert"）。
期間: J_EQ_START / J_EQ_END（既定 2016-07 / 2026-05、Standard窓内）。

実行例: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\fetch_margin_short.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402

START = get_env("J_EQ_START", "2016-07") or "2016-07"
END = get_env("J_EQ_END", "2026-05") or "2026-05"
DATASETS = [s.strip() for s in
            (get_env("J_MS_DATASETS", "ratio,weekly,positions,alert") or "").split(",")
            if s.strip()]

_START_D = max(pd.Period(START, "M").start_time, pd.Timestamp("2016-06-13"))
_END_D = min(pd.Period(END, "M").end_time, pd.Timestamp.today().normalize())


def _loop_dates(label: str, dates, fetch_fn, stepback: int = 0) -> None:
    """日付列を反復取得。stepback>0 は空なら数日遡る（週次の祝日対応）。"""
    n = len(dates)
    got = rows = 0
    t0 = time.monotonic()
    for i, d in enumerate(dates, 1):
        df = None
        for k in range(stepback + 1):
            dd = (d - pd.Timedelta(days=k)).strftime("%Y%m%d")
            try:
                df = fetch_fn(dd)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {label} {dd}: {str(e)[:80]}")
                df = None
                break
            if df is not None and not df.empty:
                break
        if df is not None and not df.empty:
            got += 1
            rows += len(df)
        if i % 50 == 0:
            el = time.monotonic() - t0
            print(f"  {label}: {i}/{n}  非空{got} 累計{rows:,}行  "
                  f"({el:.0f}s, {el / i:.2f}s/件)")
    print(f"  ★ {label} 完了: {n}日中 非空{got}, 累計{rows:,}行")


def do_ratio() -> None:
    print("\n=== 業種別空売り比率 short-ratio（S33別・全期間）===")
    master = jq.fetch_listed_info()
    s33s = sorted({str(x) for x in master.get("S33", pd.Series(dtype=object)).dropna()}
                  - {"9999", "", "nan"})
    print(f"  対象 {len(s33s)} 業種, 期間 {_START_D:%Y-%m-%d}〜{_END_D:%Y-%m-%d}")
    rows = 0
    for i, s in enumerate(s33s, 1):
        try:
            df = jq.fetch_short_ratio(s33=s, frm=_START_D.strftime("%Y%m%d"),
                                      to=_END_D.strftime("%Y%m%d"))
            rows += len(df)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] S33={s}: {str(e)[:80]}")
    print(f"  ★ short-ratio 完了: {len(s33s)}業種, 累計{rows:,}行")


def do_weekly() -> None:
    print("\n=== 信用取引週末残高 weekly margin-interest（毎週金曜）===")
    fridays = pd.date_range(_START_D, _END_D, freq="W-FRI")
    print(f"  {len(fridays)} 週, 期間 {_START_D:%Y-%m-%d}〜{_END_D:%Y-%m-%d}")
    _loop_dates("weekly", fridays, lambda dd: jq.fetch_weekly_margin(date=dd), stepback=3)


def do_positions() -> None:
    print("\n=== 空売り残高報告 short-sale-report（各営業日 calc_date）===")
    bdays = pd.bdate_range(_START_D, _END_D)
    print(f"  {len(bdays)} 営業日")
    _loop_dates("positions", bdays, lambda dd: jq.fetch_short_positions(calc_date=dd))


def do_alert() -> None:
    print("\n=== 日々公表信用取引残高 margin-alert（各営業日）===")
    bdays = pd.bdate_range(_START_D, _END_D)
    print(f"  {len(bdays)} 営業日")
    _loop_dates("alert", bdays, lambda dd: jq.fetch_margin_alert(date=dd))


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"信用・空売り一括取得  期間 {START}〜{END}  対象={DATASETS}")
    print(f"レート間隔={jq._MIN_INTERVAL}s（Standardは0.7s推奨）")
    runners = {"ratio": do_ratio, "weekly": do_weekly,
               "positions": do_positions, "alert": do_alert}
    t0 = time.monotonic()
    for ds in DATASETS:
        if ds in runners:
            runners[ds]()
        else:
            print(f"[skip] 未知のデータセット: {ds}")
    print(f"\n全完了: {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
