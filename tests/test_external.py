"""外部クロスアセット/マクロ正準ローダ（external.py）の検証。ネットワーク不要・合成parquet。"""
import numpy as np
import pandas as pd

from invest_system.data.external import (
    _PRICE_FILES,
    asof_align,
    list_external,
    load_external_prices,
    load_macro,
)


def _mk_price(invdir, key, dates, close):
    pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.nan, "change_pct": 0.0},
        index=pd.DatetimeIndex(dates, name="date"),
    ).to_parquet(invdir / _PRICE_FILES[key])


def _mk_macro(supdir, dates, **cols):
    pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date")).to_parquet(
        supdir / "macro_extended.parquet")


def test_load_external_prices_wide_and_keys(tmp_path):
    inv = tmp_path / "investers"; inv.mkdir()
    d = pd.date_range("2024-01-01", periods=5, freq="D")
    _mk_price(inv, "gold", d, [2000., 2010, 2020, 2030, 2040])
    _mk_price(inv, "usdjpy", d, [150., 151, 152, 153, 154])
    px = load_external_prices(["gold", "usdjpy"], field="close", base=str(tmp_path))
    assert list(px.columns) == ["gold", "usdjpy"]      # 正準キー・順序保持
    assert px.loc["2024-01-03", "gold"] == 2020.0
    one = load_external_prices(["usdjpy"], field="open", base=str(tmp_path))
    assert list(one.columns) == ["usdjpy"] and len(one) == 5


def test_load_macro_dedupe_and_rename(tmp_path):
    sup = tmp_path / "supplemental"; sup.mkdir()
    d = pd.date_range("2024-01-01", periods=3, freq="D")
    _mk_macro(sup, d, vix=[14., 15, 16], vix_yf=[14., 15, 16], vix_dup4=[14., 15, 16],
              japan_cpi=[100., 100, 100], gbp_jpy=[190., 191, 192],
              sp500=[5000., 5010, 5020])           # 価格＝マクロから除外されるべき
    m = load_macro(base=str(tmp_path))
    assert "vix" in m.columns and "vix_yf" not in m.columns       # VIX重複正規化
    assert "jp_cpi" in m.columns and "japan_cpi" not in m.columns  # 正準名へリネーム
    assert "gbpjpy" in m.columns
    assert "sp500" not in m.columns                               # 価格は除外


def test_asof_align_no_lookahead_and_ffill(tmp_path):
    cal = pd.date_range("2024-01-01", periods=12, freq="D")
    s = pd.Series(np.arange(12, dtype=float), index=cal, name="vix")
    s.iloc[5] = np.nan                              # 01-06 を休日欠損に
    rebal = pd.to_datetime(["2024-01-05", "2024-01-07", "2024-01-09"])
    a = asof_align(s, rebal, lag_days=1)
    assert a.loc["2024-01-05", "vix"] == 3.0        # cutoff 01-04 → 3
    assert a.loc["2024-01-07", "vix"] == 4.0        # cutoff 01-06(欠損)→ffillで 01-05=4
    # 先読みなし：未来改変は ≤t 出力に影響しない
    s2 = s.copy(); s2.iloc[7:] = 999.0
    a2 = asof_align(s2, rebal, lag_days=1)
    assert a2.loc["2024-01-05", "vix"] == 3.0       # 過去は不変
    assert a2.loc["2024-01-09", "vix"] == 999.0     # cutoff 01-08 は改変を反映


def test_list_external_marks_missing(tmp_path):
    inv = tmp_path / "investers"; inv.mkdir()
    d = pd.date_range("2024-01-01", periods=4, freq="D")
    _mk_price(inv, "gold", d, [1., 2, 3, 4])
    le = list_external(base=str(tmp_path))
    g = le[le["key"] == "gold"].iloc[0]
    assert g["kind"] == "price" and g["class"] == "metal" and g["rows"] == 4
    assert (le[le["key"] == "wti"]["source"] == "MISSING").all()   # 未取得は可視化
