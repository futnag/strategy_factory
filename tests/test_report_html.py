"""HTML判定レポート生成の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd

from invest_system.research.data_view import AsOfView
from invest_system.research.judge import judge_grid
from invest_system.research.report_html import to_html, write_html
from invest_system.research.strategy import CrossSectionalStrategy
from invest_system.validation.registry import TrialRegistry


def _verdict():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2016-01-31", periods=80, freq="ME")
    codes = [f"S{i}" for i in range(30)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.05, (80, 30)), axis=0),
                         index=idx, columns=codes)
    view = AsOfView({"close": close})
    strategies = [
        CrossSectionalStrategy(
            pd.DataFrame(rng.normal(0, 1, (80, 30)), index=idx, columns=codes),
            name=f"noise{i}") for i in range(3)]
    with TrialRegistry(":memory:") as reg:
        return judge_grid(strategies, view, scope="html_test",
                          hypothesis="pure noise has no edge",
                          economic_rationale="randomly generated factors",
                          registry=reg, costs_bps=0.0)


def test_to_html_contains_core_elements():
    v = _verdict()
    h = to_html(v)
    assert h.startswith("<!doctype html>")
    assert "html_test" in h                  # scope
    assert "<svg" in h                        # エクイティ曲線（SVG）
    assert "FAIL" in h or "PASS" in h         # 判定バナー
    assert "noise0" in h                      # 戦略名
    assert "DSR" in h


def test_write_html(tmp_path):
    v = _verdict()
    p = write_html(v, str(tmp_path / "r.html"))
    assert p.endswith("r.html")
    text = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "判定レポート" in text and "<table" in text
