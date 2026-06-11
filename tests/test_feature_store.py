"""Features (Gold) 層 feature_store.py の検証。ネットワーク不要・合成 Silver。"""
import numpy as np
import pandas as pd

from invest_system.data.feature_store import (
    build_price_features,
    load_feature,
    materialize_features,
)


def _silver_adjclose(tmp, dates, data):
    wd = tmp / "processed" / "equities" / "wide"
    wd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="Date")).to_parquet(
        wd / "adj_close.parquet")


def test_price_features_values_and_history(tmp_path):
    d = pd.date_range("2022-01-03", periods=300, freq="B")
    rng = np.random.default_rng(0)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, (300, 2)), axis=0)
    _silver_adjclose(tmp_path, d, {"A": px[:, 0], "B": px[:, 1]})
    rep = build_price_features(base=str(tmp_path))
    assert rep["returns"] == [300, 2]
    ret = load_feature("returns", base=str(tmp_path))
    assert abs(float(ret.iloc[1]["A"]) - (px[1, 0] / px[0, 0] - 1)) < 1e-4
    mom = load_feature("momentum_12_1", base=str(tmp_path))
    assert np.isnan(mom.iloc[10]["A"])               # 252日履歴未満は NaN
    assert not np.isnan(mom.iloc[-1]["A"])           # 末尾は計算可
    rev = load_feature("reversal_5", base=str(tmp_path))
    assert abs(float(rev.iloc[6]["A"]) + (px[6, 0] / px[1, 0] - 1)) < 1e-4  # 符号反転


def test_price_features_no_lookahead(tmp_path):
    d = pd.date_range("2022-01-03", periods=120, freq="B")
    rng = np.random.default_rng(1)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, (120, 2)), axis=0)
    _silver_adjclose(tmp_path, d, {"A": px[:, 0], "B": px[:, 1]})
    build_price_features(base=str(tmp_path), vol_window=10)
    ret1 = load_feature("returns", base=str(tmp_path))
    vol1 = load_feature("vol_10", base=str(tmp_path))
    k = 80
    px2 = px.copy(); px2[k + 1:] *= 1.5              # 未来を改変
    _silver_adjclose(tmp_path, d, {"A": px2[:, 0], "B": px2[:, 1]})
    build_price_features(base=str(tmp_path), vol_window=10)
    pd.testing.assert_frame_equal(
        ret1.iloc[:k + 1], load_feature("returns", base=str(tmp_path)).iloc[:k + 1])
    pd.testing.assert_frame_equal(
        vol1.iloc[:k + 1], load_feature("vol_10", base=str(tmp_path)).iloc[:k + 1])


def _silver_field(tmp, name, dates, data):
    wd = tmp / "processed" / "equities" / "wide"
    wd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="Date")).to_parquet(
        wd / f"{name}.parquet")


def test_mask_non_tradable_excludes_pinned_prices(tmp_path):
    # mask-first（C3）：ストップ高引けの価格は特徴量計算に入れない。
    d = pd.date_range("2022-01-03", periods=40, freq="B")
    n, k = len(d), 20                                # day k: A がストップ高引け
    a = np.linspace(100.0, 139.0, n)
    b = np.full(n, 50.0)
    close = {"A": a, "B": b}
    high = {"A": a + 1.0, "B": b + 1.0}
    ul = {"A": np.zeros(n), "B": np.zeros(n)}
    high["A"] = high["A"].copy(); high["A"][k] = a[k]      # close>=high＝張り付き
    ul["A"] = ul["A"].copy(); ul["A"][k] = 1.0
    _silver_field(tmp_path, "adj_close", d, close)
    _silver_field(tmp_path, "close", d, close)
    _silver_field(tmp_path, "high", d, high)
    _silver_field(tmp_path, "low", d, {"A": a - 1.0, "B": b - 1.0})
    _silver_field(tmp_path, "upper_limit", d, ul)
    _silver_field(tmp_path, "lower_limit", d, {"A": np.zeros(n), "B": np.zeros(n)})
    _silver_field(tmp_path, "volume", d, {"A": np.full(n, 100.0),
                                          "B": np.full(n, 100.0)})
    build_price_features(base=str(tmp_path), mask_non_tradable=True)
    ret = load_feature("returns", base=str(tmp_path))
    assert np.isnan(ret.iloc[k]["A"]) and np.isnan(ret.iloc[k + 1]["A"])
    assert not np.isnan(ret.iloc[k]["B"])              # 他銘柄・他日は不変
    assert not np.isnan(ret.iloc[k + 2]["A"])
    build_price_features(base=str(tmp_path))           # 既定はマスク無し＝従来どおり
    assert not np.isnan(load_feature("returns", base=str(tmp_path)).iloc[k]["A"])


def test_regime_pit_and_labels(tmp_path):
    d = pd.date_range("2020-01-02", periods=600, freq="B")
    rng = np.random.default_rng(2)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, (600, 5)), axis=0)
    _silver_adjclose(tmp_path, d, {f"S{i}": px[:, i] for i in range(5)})
    materialize_features(base=str(tmp_path))
    reg = load_feature("regime", base=str(tmp_path))
    assert {"mkt_ret", "mkt_vol", "vol_pct", "vol_regime", "trend_up"}.issubset(reg.columns)
    assert reg["vol_regime"].dropna().isin([0, 1, 2]).all()
    assert reg["trend_up"].dropna().isin([0, 1]).all()
    reg_early = reg.iloc[:300].copy()
    px2 = px.copy(); px2[350:] *= 1.3                # 未来を改変
    _silver_adjclose(tmp_path, d, {f"S{i}": px2[:, i] for i in range(5)})
    materialize_features(base=str(tmp_path))
    # ≤t のレジーム（拡張窓 percentile・trailing MA）は未来改変に不変
    pd.testing.assert_frame_equal(
        reg_early, load_feature("regime", base=str(tmp_path)).iloc[:300])
