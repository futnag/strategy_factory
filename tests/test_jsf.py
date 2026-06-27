"""JSF 貸借取引パーサ・借株コスト・需給シグナルのオフライン検証（ネット不要）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invest_system.data.sources import jsf  # noqa: E402

_CSV = (
    "日本証券金融 貸借取引残高（2026/06/26）\n"          # タイトル行（ヘッダ前）
    "日付,銘柄コード,銘柄名,融資残高,貸株残高,差引,逆日歩,規制措置\n"
    "20260626,6920,レーザーテック,\"1,234,500\",\"2,345,600\",\"-1,111,100\",15.50,\n"
    "20260626,7203,トヨタ自動車,\"500,000\",\"100,000\",\"400,000\",0,\n"
    "20260626,285A0,キオクシア,\"1,000\",\"9,000\",\"-8,000\",30.00,貸株注意喚起\n"
)


def test_parse_basic_schema_and_normalization():
    df = jsf.parse_jsf_csv(_CSV)
    assert list(df.columns) == jsf.CANON_COLUMNS
    assert len(df) == 3
    # 4桁→5桁正規化 / 英数字コードはそのまま
    assert set(df["Code"]) == {"69200", "72030", "285A0"}
    row = df.set_index("Code").loc["69200"]
    assert row["loan_balance"] == 1_234_500.0
    assert row["lending_balance"] == 2_345_600.0
    assert row["premium_rate"] == 15.50
    assert df.set_index("Code").loc["285A0", "regulation"] == "貸株注意喚起"
    assert df["Date"].iloc[0] == pd.Timestamp("2026-06-26")
    assert (df["source"] == "jsf_manual").all()


def test_parse_tsv_variant():
    tsv = _CSV.replace(",", "\t")
    # 数値内のカンマはタブ化で壊れるため、タブ専用の最小データで検証
    tsv = ("日付\t銘柄コード\t融資残高\t貸株残高\t差引\t逆日歩\t規制措置\n"
           "20260626\t6920\t1234500\t2345600\t-1111100\t15.5\t\n")
    df = jsf.parse_jsf_csv(tsv)
    assert len(df) == 1 and df["Code"].iloc[0] == "69200"
    assert df["premium_rate"].iloc[0] == 15.5


def test_net_balance_filled_when_missing():
    csv = ("銘柄コード,融資残高,貸株残高,逆日歩\n"
           "6861,300,100,0\n")
    df = jsf.parse_jsf_csv(csv, default_date="20260626")
    assert df["net_balance"].iloc[0] == 200.0          # 融資300 − 貸株100
    assert df["Date"].iloc[0] == pd.Timestamp("2026-06-26")


def test_garbage_returns_empty():
    assert jsf.parse_jsf_csv("これはヘッダの無いゴミ\n1,2,3\n").empty
    assert jsf.parse_jsf_csv(b"").empty


def test_borrow_cost_bps_panel():
    df = jsf.parse_jsf_csv(_CSV)
    close = pd.DataFrame(
        {"69200": [30000.0], "72030": [3000.0], "285A0": [1500.0]},
        index=[pd.Timestamp("2026-06-26")])
    bps = jsf.borrow_cost_bps_panel(df, close, base_bps=115.0, ann_days=245)
    # 逆日歩15.5円/30000円 × 245日 × 1e4 ＋ 115 ≈ 1265.8 + 115
    assert abs(bps.loc[pd.Timestamp("2026-06-26"), "69200"] - (15.5 / 30000 * 245 * 1e4 + 115)) < 1e-6
    # 逆日歩0 → 床のみ
    assert bps.loc[pd.Timestamp("2026-06-26"), "72030"] == 115.0
    # 高逆日歩は床を大きく上回る
    assert bps.loc[pd.Timestamp("2026-06-26"), "285A0"] > 4000.0


def test_loan_lending_ratio_and_squeeze():
    df = jsf.parse_jsf_csv(_CSV)
    ratio = jsf.loan_lending_ratio(df)
    # 6920: 融資1,234,500 / 貸株2,345,600 < 1（貸株超過＝クラウディング）
    assert ratio.loc[pd.Timestamp("2026-06-26"), "69200"] < 1.0
    sq = jsf.squeeze_flags(df)
    assert bool(sq.loc[pd.Timestamp("2026-06-26"), "285A0"]) is True   # 逆日歩>0
    assert bool(sq.loc[pd.Timestamp("2026-06-26"), "72030"]) is False  # 逆日歩=0


def test_ingest_dir_and_load_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "20260626_jsf.csv").write_text(_CSV, encoding="utf-8")
    cache = tmp_path / "cache"
    rep = jsf.ingest_dir(str(raw), str(cache), glob="*.csv", verbose=False)
    assert rep["written"] == 1 and rep["rows"] == 3
    # 冪等：再実行でスキップ
    rep2 = jsf.ingest_dir(str(raw), str(cache), glob="*.csv", verbose=False)
    assert rep2["written"] == 0 and rep2["skipped"] == 1
    out = jsf.load_jsf(str(cache))
    assert len(out) == 3 and set(out["Code"]) == {"69200", "72030", "285A0"}


def test_load_missing_dir_is_empty(tmp_path):
    assert jsf.load_jsf(str(tmp_path / "nope")).empty
