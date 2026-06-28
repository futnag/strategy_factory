"""所有構造オーバーレイ（高外国人除外・PEADスリーブ用）のオフライン検証。docs/50。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invest_system.equities import ownership as ow  # noqa: E402


def _sig():
    return pd.DataFrame({"A": [1.0, 2.0], "B": [0.5, 1.5], "C": [-1.0, 0.0]},
                        index=pd.to_datetime(["2024-01-31", "2024-02-29"]))


def _own():
    # foreign%: A=60(高), B=30, C=5。cutoff_q=0.67 → thr≈40 → A のみ除外。
    return pd.DataFrame({"Code": ["A", "B", "C"], "foreign_pct": [60.0, 30.0, 5.0],
                         "individual_pct": [10.0, 30.0, 70.0]})


def test_disabled_is_noop():
    s = _sig()
    pd.testing.assert_frame_equal(ow.apply_high_foreign_exclusion(s, _own(), enabled=False), s)


def test_exclude_high_foreign():
    s = _sig()
    out = ow.apply_high_foreign_exclusion(s, _own(), enabled=True, cutoff_q=0.67, mode="exclude")
    assert out["A"].isna().all()                 # 高外国人 A は除外
    assert out["B"].equals(s["B"]) and out["C"].equals(s["C"])  # 他は不変


def test_downweight():
    s = _sig()
    z = ow.apply_high_foreign_exclusion(s, _own(), enabled=True, cutoff_q=0.67,
                                        mode="downweight", shrink=0.0)
    assert (z["A"] == 0.0).all()
    h = ow.apply_high_foreign_exclusion(s, _own(), enabled=True, cutoff_q=0.67,
                                        mode="downweight", shrink=0.5)
    assert np.allclose(h["A"].values, s["A"].values * 0.5)


def test_missing_ownership_is_noop():
    s = _sig()
    empty = pd.DataFrame(columns=["Code", "foreign_pct", "individual_pct"])
    pd.testing.assert_frame_equal(ow.apply_high_foreign_exclusion(s, empty, enabled=True), s)


def test_unknown_code_not_excluded():
    s = _sig()
    s["D"] = [3.0, 4.0]                           # D は所有データに無い
    out = ow.apply_high_foreign_exclusion(s, _own(), enabled=True, cutoff_q=0.67)
    assert out["D"].equals(s["D"])               # unknown は保持（graceful）


def test_high_foreign_codes_cutoff():
    hi = ow.high_foreign_codes(_own(), cutoff_q=0.67)
    assert "A" in hi and "B" not in hi and "C" not in hi


def test_load_missing_file(tmp_path):
    assert ow.load_ownership(str(tmp_path / "nope.parquet")).empty
