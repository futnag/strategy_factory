"""Phase 2 注文生成（単元未満株・指数ヘッジ置換・ロット丸め）の検証。ネット不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.production import equity_orders, hedge_contracts, lot_orders


def test_equity_orders_fractional_longs_and_short_aggregation():
    w = pd.Series({"A": 0.4, "B": 0.1, "C": -0.3, "D": -0.2, "E": 0.001})
    px = pd.Series({"A": 2000.0, "B": 50000.0, "C": 1000.0, "D": 1.0,
                    "E": 9000.0})
    orders, short = equity_orders(w, px, capital=600_000, lot=1)
    o = orders.set_index("code")
    assert o.loc["A", "shares"] == 120          # 24万円 / 2000円
    assert o.loc["B", "shares"] == 1            # 6万円 / 5万円 → 1株
    assert "E" not in o.index                   # 600円 < 9000円 → 1株も買えず脱落
    assert "C" not in o.index and "D" not in o.index   # ショートはロング表に出ない
    assert short == pytest.approx(0.5 * 600_000)        # |−0.3|+|−0.2| を集約
    assert (o["yen"] == o["shares"] * o["price"]).all()


def test_equity_orders_missing_price_dropped():
    w = pd.Series({"A": 0.5, "B": 0.5})
    px = pd.Series({"A": 1000.0, "B": np.nan})
    orders, _ = equity_orders(w, px, capital=100_000)
    assert list(orders["code"]) == ["A"]


def test_hedge_contracts_rounding():
    # 想定元本30万円 / マイクロ1枚=4万×10=40万円 → 1枚（最近接）
    n, yen = hedge_contracts(300_000, index_price=40_000.0, multiplier=10.0)
    assert n == 1 and yen == pytest.approx(400_000)
    n2, yen2 = hedge_contracts(150_000, 40_000.0)        # 0.375枚 → 0枚
    assert n2 == 0 and yen2 == 0.0
    assert hedge_contracts(0.0, 40_000.0) == (0, 0.0)


def test_lot_orders_steps_and_paper_passthrough():
    w = pd.Series({"nk225_fut": 0.55, "gold": 0.30})
    px = pd.Series({"nk225_fut": 40_000.0, "gold": 16_000.0})
    out = lot_orders(w, px, capital=300_000,
                     lot_units={"nk225_fut": 10.0},      # gold は表に無い＝ペーパー保持
                     lot_steps={"nk225_fut": 1}).set_index("asset")
    # 16.5万円 / 40万円 → 0.41枚 → 0枚（刻み1）＝ショートフォール全額
    assert out.loc["nk225_fut", "lots"] == 0
    assert out.loc["nk225_fut", "shortfall_yen"] == pytest.approx(165_000)
    # ロット表に無い資産は想定元本のまま（ペーパー）＝ショートフォール0
    assert np.isnan(out.loc["gold", "lots"])
    assert out.loc["gold", "filled_yen"] == pytest.approx(90_000)
    assert out.loc["gold", "shortfall_yen"] == 0.0


def test_lot_orders_cfd_step():
    w = pd.Series({"sp500": 0.5})
    px = pd.Series({"sp500": 900_000.0})                 # 1ロット=指数×1（円換算済想定）
    out = lot_orders(w, px, capital=300_000, lot_units={"sp500": 1.0},
                     lot_steps={"sp500": 0.1}).set_index("asset")
    assert out.loc["sp500", "lots"] == pytest.approx(0.2)   # 15万/90万=0.167→0.2
    assert out.loc["sp500", "filled_yen"] == pytest.approx(180_000)
