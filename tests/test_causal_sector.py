"""セクター因果レジームモジュールのユニットテスト。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.causal_sector.config import (
    ALL_SECTORS, SECTOR_PROFILES, collider_candidates, REPRESENTATIVE_SECTORS,
)
from invest_system.research.causal_sector.meta import feature_columns_all33
from invest_system.research.causal_sector.graph import (
    CausalGraphResult, _identify_colliders_from_graph,
)
from invest_system.research.causal_sector.regime import page_cusum


def test_sector_profiles_cover_representatives():
    for s33 in REPRESENTATIVE_SECTORS:
        assert s33 in SECTOR_PROFILES
        p = SECTOR_PROFILES[s33]
        assert len(p.drivers) >= 4
        assert p.kind in ("heavy_industry", "it_comm", "financial", "energy_util", "default")


def test_collider_candidates_includes_watch():
    prof = SECTOR_PROFILES["5250"]
    cands = collider_candidates(["RET", "VALUE", "VOL", "MOM"], prof)
    assert "VOL" in cands


def test_identify_colliders_from_graph():
    # 3 vars: VALUE, RET, VOL; both VALUE and RET point to VOL
    var_names = ["VALUE", "RET", "VOL"]
    graph = np.zeros((3, 3, 2), dtype="<U3")
    graph[0, 2, 0] = "-->"   # VALUE -> VOL
    graph[1, 2, 0] = "-->"   # RET -> VOL
    val = np.zeros((3, 3, 2))
    coll = _identify_colliders_from_graph(var_names, graph, val, 1, "VALUE", "RET")
    assert "VOL" in coll


def test_page_cusum_detects_shift():
    rng = np.random.default_rng(0)
    x = pd.Series(np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)]))
    alarms = page_cusum(x, min_periods=20, h_sd=3.0)
    assert len(alarms) >= 1


def test_all_sectors_count():
    assert len(ALL_SECTORS) == 33
    assert "9999" not in ALL_SECTORS
    assert "5250" in ALL_SECTORS


def test_feature_columns_all33():
    cols = feature_columns_all33()
    assert len(cols) == 8
    assert "edge_heavy_industry" in cols
    assert "edge_it_comm" in cols
    assert "edge_financial" in cols


def test_causal_graph_result_dataclass():
    r = CausalGraphResult(
        var_names=["VALUE", "RET"], val_matrix=np.zeros((2, 2, 2)),
        graph=np.zeros((2, 2, 2), dtype="<U3"), tau_max=1,
        colliders=["VOL"], adjustment_set={"VALUE_to_RET": ["MOM"]},
    )
    assert r.colliders == ["VOL"]
    assert r.adjustment_set["VALUE_to_RET"] == ["MOM"]