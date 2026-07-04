"""loop_lint の回帰テスト。

L6 の tz 混在バグ（2026-07-04 incident・P-7）: operator が書くロックの started_at が
tz 付き（date -Iseconds）だと datetime.now()（naive）との減算で TypeError となり、
loop_lint が HARD 誤発火＝headless 運用が毎回停止していた。
"""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "loop_lint", Path(__file__).resolve().parent.parent / "examples" / "loop_lint.py")
loop_lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop_lint)


def test_lock_age_naive():
    """tz 無し（datetime.now().isoformat() 相当）で例外なく経過時間を返す。"""
    ts = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
    assert loop_lint.lock_age_hours(ts) == pytest.approx(2.0, abs=0.05)


def test_lock_age_tz_aware():
    """tz 付き（date -Iseconds 相当）でも TypeError にならず妥当な値を返す。"""
    ts = (datetime.now(timezone(timedelta(hours=9))) - timedelta(hours=3)) \
        .isoformat(timespec="seconds")
    assert loop_lint.lock_age_hours(ts) == pytest.approx(3.0, abs=0.05)


def test_lock_age_fresh():
    """取得直後（0h 付近）＝並行起動検知の下限。tz 付きでも動く。"""
    ts = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    assert loop_lint.lock_age_hours(ts) < 0.1
