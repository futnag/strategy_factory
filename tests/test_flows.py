"""投資部門別フロー分析層（純関数）の検証。ネットワーク不要。"""
import pandas as pd
import pytest

from invest_system.equities.flows import (
    net_flow_intensity, section_net_flow,
)


def _df():
    return pd.DataFrame({
        "EnDate": pd.to_datetime(["2026-04-24", "2026-05-01", "2026-04-24"]),
        "Section": ["TokyoNagoya", "TokyoNagoya", "TSEPrime"],
        "FrgnBal": [500.0, -200.0, 999.0],
        "FrgnTot": [1000.0, 800.0, 5000.0],
    })


def test_section_net_flow_selects_section_and_sorts():
    s = section_net_flow(_df(), investor="foreign", section="TokyoNagoya")
    assert list(s.index) == [pd.Timestamp("2026-04-24"), pd.Timestamp("2026-05-01")]
    assert s.iloc[0] == 500.0 and s.iloc[1] == -200.0   # TSEPrime行は除外


def test_net_flow_intensity_normalizes():
    s = net_flow_intensity(_df(), investor="foreign", section="TokyoNagoya")
    assert s.iloc[0] == pytest.approx(0.5)              # 500/1000
    assert s.iloc[1] == pytest.approx(-0.25)            # -200/800


def test_empty_or_missing_returns_empty():
    assert section_net_flow(pd.DataFrame()).empty
    assert net_flow_intensity(pd.DataFrame()).empty
    # 主体列が無いケース
    assert section_net_flow(pd.DataFrame({"Section": ["X"], "EnDate": [pd.NaT]})).empty
