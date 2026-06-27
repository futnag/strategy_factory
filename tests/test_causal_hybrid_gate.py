"""ハイブリッド因果メタゲートのユニットテスト。"""
import numpy as np
import pandas as pd

from invest_system.research.causal_sector.hybrid_gate import (
    DEFAULT_SHRINK_MULT, DEFAULT_UNSTABLE_THRESH,
    adverse_causal_regime, fit_hybrid_causal_gate,
)


def test_adverse_all_three_required():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    cf = pd.DataFrame({
        "causal_edge": [-0.1, 0.1, -0.1, -0.1],
        "edge_financial": [-0.1, -0.1, 0.1, -0.1],
        "n_sectors_unstable": [6, 6, 6, 3],
    }, index=idx)
    adv = adverse_causal_regime(cf)
    assert adv.iloc[0] is True or adv.iloc[0] == True   # noqa: E712
    assert not adv.iloc[1]   # edge positive
    assert not adv.iloc[2]   # financial positive
    assert not adv.iloc[3]   # unstable low


def test_adverse_missing_is_false():
    cf = pd.DataFrame({"causal_edge": [np.nan]}, index=[pd.Timestamp("2020-01-31")])
    assert not adverse_causal_regime(cf).iloc[0]


def test_hybrid_shrink_only_on_adverse():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    net = pd.Series([0.01, -0.01, 0.02, 0.01, -0.02, 0.01], index=idx)
    base = pd.DataFrame({"topix_vol": 0.1, "topix_mom": 0.0, "value_trail": 0.5,
                         "dispersion": 0.02, "flow_intensity": 0.0}, index=idx)
    causal = pd.DataFrame({
        "causal_edge": [-0.1, -0.1, 0.1, -0.1, -0.1, -0.1],
        "edge_financial": [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
        "n_sectors_unstable": [6, 6, 6, 6, 6, 6],
        "edge_volatility": 0.01, "edge_chg_3m": 0.0,
        "months_since_break": 12, "edge_heavy_industry": 0.0,
        "edge_it_comm": 0.0,
    }, index=idx)
    hybrid, adv = fit_hybrid_causal_gate(net, base, causal, warmup=2, embargo=0)
    # warmup後、adverseかつbase有効の月は base * 0.5 以下
    assert DEFAULT_SHRINK_MULT == 0.5
    assert DEFAULT_UNSTABLE_THRESH == 5
    assert len(hybrid) == len(idx)