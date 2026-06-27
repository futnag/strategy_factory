"""因果レジーム月次監視のユニットテスト。"""
import json

import numpy as np
import pandas as pd

from invest_system.research.causal_sector.monitor import (
    AdverseStatus,
    compute_adverse_status,
    format_monitor_md,
    snapshot_to_dict,
    _sector_table_at,
    _status_from_adverse,
)


def test_compute_adverse_status_all_met():
    row = pd.Series({
        "causal_edge": -0.05,
        "edge_financial": -0.02,
        "n_sectors_unstable": 6,
    })
    adv = compute_adverse_status(row)
    assert adv.all_met
    assert not adv.missing
    assert _status_from_adverse(adv) == "ADVERSE"


def test_compute_adverse_status_missing():
    row = pd.Series({"causal_edge": np.nan, "edge_financial": -0.1,
                     "n_sectors_unstable": 8})
    adv = compute_adverse_status(row)
    assert adv.missing
    assert not adv.all_met
    assert _status_from_adverse(adv) == "CAUTION"


def test_compute_adverse_status_caution_two_of_three():
    row = pd.Series({
        "causal_edge": -0.05,
        "edge_financial": 0.02,
        "n_sectors_unstable": 6,
    })
    adv = compute_adverse_status(row)
    assert not adv.all_met
    assert adv.n_conditions_met == 2
    assert _status_from_adverse(adv) == "CAUTION"


def test_sector_table_unstable_flag():
    idx = pd.to_datetime(["2024-01-31", "2024-01-31", "2024-01-31"])
    long_m = pd.DataFrame({
        "s33": ["5250", "7050", "3600"],
        "kind": ["it_comm", "financial", "heavy_industry"],
        "causal_edge": [-0.1, 0.05, 0.02],
        "edge_volatility": [0.05, 0.02, 0.08],
        "edge_chg_3m": [0.0, 0.0, 0.0],
        "months_since_break": [12, 12, 12],
    }, index=idx)
    tbl = _sector_table_at(long_m, pd.Timestamp("2024-01-31"))
    assert len(tbl) == 3
    assert tbl.loc[tbl["s33"] == "3600", "unstable"].iloc[0]


def test_snapshot_to_dict_json_serializable():
    adv = AdverseStatus(True, True, True, True, False)
    snap_like = type("S", (), {})()
    # minimal mock via building dict path — use format on real components
    payload = {
        "adverse": adv,
        "aggregate": {"causal_edge": -0.1},
    }
    d = snapshot_to_dict(type("Snap", (), {
        "asof": pd.Timestamp("2024-06-30"),
        "generated_at": pd.Timestamp("2024-07-01", tz="UTC").to_pydatetime(),
        "status": "ADVERSE",
        "aggregate": payload["aggregate"],
        "adverse": adv,
        "sector_edges": pd.DataFrame(),
        "recent_history": pd.DataFrame(),
        "n_sectors": 0,
        "cache_path": None,
        "notes": [],
    })())
    json.dumps(d)
    assert d["status"] == "ADVERSE"
    assert d["hybrid_gate_reference"]["would_shrink"] is True


def test_format_monitor_md_contains_status():
    adv = AdverseStatus(False, False, False, False, False)
    md = format_monitor_md(type("Snap", (), {
        "asof": pd.Timestamp("2024-06-30"),
        "generated_at": pd.Timestamp("2024-07-01", tz="UTC").to_pydatetime(),
        "status": "NORMAL",
        "aggregate": {"causal_edge": 0.05, "edge_financial": 0.02,
                      "n_sectors_unstable": 2},
        "adverse": adv,
        "sector_edges": pd.DataFrame(),
        "recent_history": pd.DataFrame(),
        "n_sectors": 0,
    })())
    assert "NORMAL" in md
    assert "監視のみ" in md