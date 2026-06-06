"""差分更新エンジン（要件4）：新規データだけ取得しローカルを最新に保つ。

各データセットについて (1) 既存キャッシュ＋マニフェストから「取得済み日」を把握し、
(2) 期間内の候補日（日次=営業日, 週次=金曜）から取得済みを除いた「欠損日」だけを
取得する。空（祝日・未公表）もマーカーでキャッシュ済みなので再取得しない＝冪等・再開
可能。純関数（candidate_dates/missing_dates）はネットワーク不要でテスト可能。

注：祝日は営業日候補に混ざるが、取得結果が空でもマーカーが残り次回スキップされるため
正しく動く（取引カレンダー依存を避けた頑健設計）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .catalog import DATASETS, REFRESH_DATASETS, Dataset
from .sources import jquants as jq


def candidate_dates(cadence: str, start, end) -> list[str]:
    """期間内の取得候補日（YYYYMMDD）。日次=平日, 週次=金曜。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e < s:
        return []
    if cadence == "weekly":
        rng = pd.date_range(s, e, freq="W-FRI")
    else:
        rng = pd.bdate_range(s, e)
    return [d.strftime("%Y%m%d") for d in rng]


def missing_dates(candidates: list[str], fetched) -> list[str]:
    """候補から取得済みを除いた欠損日。"""
    f = set(fetched)
    return [d for d in candidates if d not in f]


def scan_cache_dates(dataset: Dataset, base: Path) -> set[str]:
    """キャッシュ dir のファイル名から取得済み日を復元（マニフェスト無しでも機能）。"""
    d = base / dataset.cache_subdir
    out: set[str] = set()
    if d.exists():
        for p in d.glob("*.parquet"):
            dt = dataset.stem_to_date(p.stem)
            if dt:
                out.add(dt)
    return out


class Manifest:
    """取得済み日の台帳（JSON）。キャッシュscanと併用する。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, list[str]] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def fetched(self, name: str) -> set[str]:
        return set(self.data.get(name, []))

    def mark(self, name: str, date: str) -> None:
        self.data.setdefault(name, [])
        if date not in self.data[name]:
            self.data[name].append(date)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False),
                             encoding="utf-8")


class DataUpdater:
    """差分更新の司令塔。"""

    def __init__(self, datasets: Optional[dict] = None,
                 refresh_datasets: Optional[dict] = None,
                 base: Optional[str] = None, manifest_path: Optional[str] = None,
                 start: str = "2016-06-13"):
        self.datasets = datasets if datasets is not None else DATASETS
        self.refresh_datasets = (refresh_datasets if refresh_datasets is not None
                                 else REFRESH_DATASETS)
        self.base = Path(base) if base else jq._CACHE
        self.manifest = Manifest(Path(manifest_path) if manifest_path
                                 else self.base / "manifest.json")
        self.start = start

    def plan(self, name: str, until) -> list[str]:
        """name の欠損日（取得予定）を返す。取得は行わない。"""
        ds = self.datasets[name]
        fetched = self.manifest.fetched(name) | scan_cache_dates(ds, self.base)
        cands = candidate_dates(ds.cadence, self.start, until)
        return missing_dates(cands, fetched)

    def update(self, names: Optional[list[str]] = None, until=None,
               verbose: bool = True) -> dict:
        """maintained データセット（または指定）を until まで最新化。

        by-date 系（欠損日のみ取得）と range-refresh 系（指数・投資部門別を全体再取得）の
        両方を対象にする。
        """
        until = pd.Timestamp(until) if until else pd.Timestamp.today().normalize()
        until_s = until.strftime("%Y-%m-%d")
        if names is None:
            names = ([n for n, d in self.datasets.items() if d.maintained]
                     + [n for n, d in self.refresh_datasets.items() if d.maintained])
        report: dict = {}
        # by-date：欠損日のみ取得
        for name in [n for n in names if n in self.datasets]:
            ds = self.datasets[name]
            miss = self.plan(name, until)
            got = 0
            for dt in miss:
                try:
                    ds.fetch_by_date(dt)
                    self.manifest.mark(name, dt)
                    got += 1
                except Exception as e:  # noqa: BLE001
                    if verbose:
                        print(f"  [warn] {name} {dt}: {str(e)[:70]}")
            self.manifest.save()
            report[name] = {"missing": len(miss), "fetched": got}
            if verbose:
                print(f"  {name}: 欠損{len(miss)} → 取得{got}")
        # range-refresh：全体を最新化（指数・投資部門別）
        for name in [n for n in names if n in self.refresh_datasets]:
            spec = self.refresh_datasets[name]
            try:
                rows = spec.refresh(self.start, until_s)
                report[name] = {"refreshed_rows": rows}
                if verbose:
                    print(f"  {name}: 再取得 {rows:,} 行")
            except Exception as e:  # noqa: BLE001
                report[name] = {"error": str(e)[:80]}
                if verbose:
                    print(f"  [warn] {name}: {str(e)[:70]}")
        return report
