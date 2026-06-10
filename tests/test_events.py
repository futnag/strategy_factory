"""イベント系シグナル（予想改訂・決算サプライズ）の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.events import (
    announcement_delay, buyback_intensity, days_to_next_announcement,
    dividend_forecast_revision, earnings_surprise, expected_announcement_month,
    forecast_revision, guidance_conservatism,
)


def test_announcement_delay_same_quarter_yoy():
    fund = pd.DataFrame({
        "Code": ["100"] * 5,
        "CurPerType": ["FY", "1Q", "FY", "1Q", "FY"],
        "DocType": ["FYFinancialStatements_Consolidated_JP",
                    "1QFinancialStatements_Consolidated_JP",
                    "FYFinancialStatements_Consolidated_JP",
                    "EarnForecastRevision",          # 臨時開示は発表日と数えない
                    "FYFinancialStatements_Consolidated_JP"],
        "DiscDate": pd.to_datetime(["2023-05-10", "2023-08-01", "2024-05-20",
                                    "2024-08-05", "2026-05-10"]),
    })
    out = announcement_delay(fund).set_index("DiscDate")["delay_days"]
    # FY: 2023-05-10 → 2024-05-20 ＝ 376日 → 遅延 +11日
    assert out.loc["2024-05-20"] == pytest.approx(11.0)
    # 1Q の2回目は EarnForecastRevision なので除外＝ペア不成立
    assert pd.Timestamp("2024-08-05") not in out.index
    # FY: 2024→2026 は 720日 ＝ 範囲外（年度ズレ）→ 採用しない
    assert pd.Timestamp("2026-05-10") not in out.index


def _fy_row(code, disc, fy_end, eps, nxf, nxt_end):
    return {"Code": code, "DiscDate": disc, "CurPerType": "FY", "EPS": eps,
            "NxFEPS": nxf, "CurFYEn": fy_end, "NxtFYEn": nxt_end}


def test_guidance_conservatism_habitual_beater():
    # 毎年「期初予想100 → 実績120」＝常習ビーター（surprise +20% が並ぶ）
    rows = [
        _fy_row("100", "2020-05-10", "2020-03-31", 110.0, 100.0, "2021-03-31"),
        _fy_row("100", "2021-05-10", "2021-03-31", 120.0, 100.0, "2022-03-31"),
        _fy_row("100", "2022-05-10", "2022-03-31", 120.0, 100.0, "2023-03-31"),
        _fy_row("100", "2023-05-10", "2023-03-31", 120.0, 100.0, "2024-03-31"),
    ]
    fund = pd.DataFrame(rows)
    fund["DiscDate"] = pd.to_datetime(fund["DiscDate"])
    out = guidance_conservatism(fund, n_years=3, min_years=2) \
        .set_index("DiscDate")["cons_score"]
    # 2021開示: surprise(2021年度実績120 vs 期初100)=+20% のみ → min_years 未満で無効
    assert pd.Timestamp("2021-05-10") not in out.index
    # 2022開示: surprise +20%, +20% → 平均 +20%
    assert out.loc["2022-05-10"] == pytest.approx(0.20)
    assert out.loc["2023-05-10"] == pytest.approx(0.20)


def test_guidance_conservatism_requires_fy_alignment():
    # 年度が飛ぶ（NxtFYEn ≠ 次行の CurFYEn）場合は surprise を作らない
    rows = [
        _fy_row("100", "2020-05-10", "2020-03-31", 100.0, 100.0, "2021-03-31"),
        _fy_row("100", "2022-05-10", "2022-03-31", 200.0, 100.0, "2023-03-31"),
        _fy_row("100", "2023-05-10", "2023-03-31", 50.0, 100.0, "2024-03-31"),
    ]
    fund = pd.DataFrame(rows)
    fund["DiscDate"] = pd.to_datetime(fund["DiscDate"])
    out = guidance_conservatism(fund, n_years=3, min_years=1) \
        .set_index("DiscDate")["cons_score"]
    # 2022開示: 前行の NxtFYEn=2021-03 ≠ CurFYEn=2022-03 → 整合せず surprise 無し
    assert pd.Timestamp("2022-05-10") not in out.index
    # 2023開示: 前行(2022)の NxtFYEn=2023-03 == CurFYEn=2023-03 → (50-100)/100 = -50%
    assert out.loc["2023-05-10"] == pytest.approx(-0.50)


def test_dividend_forecast_revision_split_adjusted():
    # 2:1 分割（係数 0.5 が 6/1 に適用）を挟み 1株配当 50→25 円：
    # 無調整なら −50% の偽減配、分割調整後は改訂率 0。
    fund = pd.DataFrame({
        "Code": ["100", "100", "100"],
        "DiscDate": pd.to_datetime(["2024-05-01", "2024-08-01", "2024-11-01"]),
        "FDivAnn": [50.0, 25.0, 30.0],
    })
    idx = pd.date_range("2024-04-01", periods=250, freq="D")
    af = pd.DataFrame(1.0, index=idx, columns=["100"])
    af.loc["2024-06-01", "100"] = 0.5
    adj_cum = af.cumprod()
    naive = dividend_forecast_revision(fund).set_index("DiscDate")["div_revision"]
    assert naive.loc["2024-08-01"] == pytest.approx(-0.50)
    adj = dividend_forecast_revision(fund, adj_cum=adj_cum) \
        .set_index("DiscDate")["div_revision"]
    assert adj.loc["2024-08-01"] == pytest.approx(0.0)       # 分割の機械的減配は中立
    assert adj.loc["2024-11-01"] == pytest.approx(0.20)      # 25→30 は真の増配 +20%


def test_buyback_intensity_split_immune_and_signs():
    fund = pd.DataFrame({
        "Code": ["100", "100", "100", "100"],
        "DiscDate": pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01",
                                    "2024-11-01"]),
        # 取得（10→20）→ 2:1分割（両方2倍＝比率不変）→ 増資（ShOut+1000）
        "TrShFY": [10.0, 20.0, 40.0, 40.0],
        "ShOutFY": [1000.0, 1000.0, 2000.0, 3000.0],
    })
    out = buyback_intensity(fund).set_index("DiscDate")["buyback"]
    assert out.loc["2024-05-01"] == pytest.approx(0.01)      # 自社株買い＝正
    assert out.loc["2024-08-01"] == pytest.approx(0.0)       # 分割は比率不変
    assert out.loc["2024-11-01"] < 0                         # 希薄化＝負


def test_forecast_revision_per_code():
    fund = pd.DataFrame({
        "Code": ["100", "100", "100", "200"],
        "DiscDate": pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01",
                                    "2024-05-01"]),
        "FEPS": [100.0, 110.0, 99.0, 50.0],
    })
    out = forecast_revision(fund).set_index(["Code", "DiscDate"])["fcst_revision"]
    # 100: 2回目 +10%、3回目 -10%。200は1開示のみ→改訂なし
    assert out.loc[("100", pd.Timestamp("2024-05-01"))] == pytest.approx(0.10)
    assert out.loc[("100", pd.Timestamp("2024-08-01"))] == pytest.approx(-0.10)
    assert ("200", pd.Timestamp("2024-05-01")) not in out.index


def test_earnings_surprise():
    fund = pd.DataFrame({
        "Code": ["100"], "DiscDate": pd.to_datetime(["2024-05-01"]),
        "EPS": [120.0], "FEPS": [100.0],
    })
    out = earnings_surprise(fund)
    assert out["surprise"].iloc[0] == pytest.approx(0.20)   # (120-100)/100


def test_expected_announcement_month():
    fund = pd.DataFrame({
        "Code": ["100", "100", "200"],
        "DiscDate": pd.to_datetime(["2023-05-15", "2024-05-15", "2023-08-10"]),
    })
    rebal = pd.to_datetime(["2024-04-30", "2024-07-31"])
    m = expected_announcement_month(fund, rebal)
    # 100は5月発表 → 4月末(翌月=5月)に発表見込みTrue、7月末(翌月=8月)はFalse
    assert bool(m.loc[pd.Timestamp("2024-04-30"), "100"]) is True
    assert bool(m.loc[pd.Timestamp("2024-07-31"), "100"]) is False
    # 200は8月発表 → 7月末(翌月=8月)にTrue
    assert bool(m.loc[pd.Timestamp("2024-07-31"), "200"]) is True


def test_days_to_next_announcement():
    fund = pd.DataFrame({
        "Code": ["100", "100"],
        "DiscDate": pd.to_datetime(["2024-02-01", "2024-05-01"]),  # 間隔90日
    })
    dates = pd.to_datetime(["2024-05-02", "2024-07-15"])
    p = days_to_next_announcement(fund, dates)["100"]
    # 次回予測=2024-05-01+90日=2024-07-30。残日数は 89 / 15。
    assert p.loc[pd.Timestamp("2024-05-02")] == pytest.approx(89)
    assert p.loc[pd.Timestamp("2024-07-15")] == pytest.approx(15)


def test_empty_inputs():
    assert forecast_revision(pd.DataFrame()).empty
    assert earnings_surprise(pd.DataFrame()).empty
    assert expected_announcement_month(pd.DataFrame(), [pd.Timestamp("2024-01-31")]).empty
    assert days_to_next_announcement(pd.DataFrame(), [pd.Timestamp("2024-01-31")]).empty
