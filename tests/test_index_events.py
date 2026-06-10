"""日経225 構成銘柄変更イベント表とウィンドウ・ヘルパーの検証。ネットワーク不要。"""
import pandas as pd
import pytest

from invest_system.equities.index_events import (
    event_holding_windows, jq_code, n225_changes, tradeable_event_legs,
    window_weights,
)


def test_table_integrity():
    df = n225_changes()
    assert len(df) >= 40
    # 発表日があるレコードは必ず発表 < 実施（PIT の前提）
    known = df.dropna(subset=["announce"])
    assert (known["announce"] < known["effective"]).all()
    # 定期見直しは発表日必須（検証済みのみ periodic を名乗れる）
    periodic_in = df[df["in_kind"] == "periodic"]
    assert periodic_in["announce"].notna().all()
    # コード形式：4文字の英数字（新形式 285A 等を含む）
    codes = pd.concat([df["out_code"].dropna(), df["in_code"].dropna()])
    assert codes.str.fullmatch(r"[0-9A-Z]{4}").all()
    # 実施日は昇順に並んでいる（公式履歴の時系列）
    assert df["effective"].is_monotonic_increasing


def test_jq_code():
    assert jq_code("7203") == "72030"
    assert jq_code("285A") == "285A0"


def test_tradeable_legs_filter_succession_and_delisting():
    adds, dels = tradeable_event_legs()
    # 継承（ARCHION 543A・NXHD 9147・しずおかFG 5831・出光 5019）は採用側に含めない
    assert not set(adds["code"]) & {"543A0", "91470", "58310", "50190"}
    # TOB 消滅（ドコモ 9437・ファミマ 8028 等）は除外側に含めない
    assert not set(dels["code"]) & {"94370", "80280", "96130"}
    # 上場継続の降格除外（東芝 6502・ニデック 6594）は除外側に残る
    assert {"65020", "65940"} <= set(dels["code"])
    # 定期の代表例（キーエンス採用・日清紡除外）
    assert "68610" in set(adds["code"]) and "31050" in set(dels["code"])


def _dates():
    return pd.DatetimeIndex(pd.date_range("2024-01-01", periods=40, freq="B"))


def test_event_holding_windows_resolves_and_drops_unknown():
    dates = _dates()
    legs = pd.DataFrame({
        "announce": [dates[5], pd.NaT],
        "effective": [dates[15], dates[20]],
        "code": ["A0001", "B0001"],
        "kind": ["periodic", "extraordinary"],
    })
    win = event_holding_windows(legs, dates, start_anchor="announce",
                                start_offset=0, end_anchor="effective",
                                end_offset=-3)
    assert list(win["code"]) == ["A0001"]          # announce 不明は脱落（PIT）
    assert win.iloc[0]["start"] == dates[5]
    assert win.iloc[0]["end"] == dates[12]
    # effective アンカーなら announce 不明でも使える（リバーサル用）
    win2 = event_holding_windows(legs, dates, start_anchor="effective",
                                 start_offset=-1, end_anchor="effective",
                                 end_offset=3)
    assert set(win2["code"]) == {"A0001", "B0001"}
    assert win2.set_index("code").loc["B0001", "start"] == dates[19]


def test_event_holding_windows_rounds_to_next_business_day():
    dates = _dates()
    sat = dates[5] + pd.Timedelta(days=5 - dates[5].weekday())  # 直近の土曜
    legs = pd.DataFrame({"announce": [sat], "effective": [dates[20]],
                         "code": ["A0001"], "kind": ["periodic"]})
    win = event_holding_windows(legs, dates, start_offset=0, end_offset=0)
    assert win.iloc[0]["start"] in dates           # 翌営業日へ丸め


def test_window_weights_equal_weight_and_neutral():
    dates = _dates()
    longs = pd.DataFrame({"code": ["A0001", "B0001"],
                          "start": [dates[2], dates[2]],
                          "end": [dates[4], dates[3]]})
    shorts = pd.DataFrame({"code": ["C0001"], "start": [dates[3]],
                           "end": [dates[5]]})
    w = window_weights(longs, shorts, dates)
    assert w[dates[2]].to_dict() == {"A0001": 0.5, "B0001": 0.5}
    assert w[dates[3]].to_dict() == {"A0001": 0.5, "B0001": 0.5, "C0001": -1.0}
    assert w[dates[4]].to_dict() == {"A0001": 1.0, "C0001": -1.0}
    assert w[dates[5]].to_dict() == {"C0001": -1.0}
    assert dates[6] not in w
    # ダラーニュートラル（両脚あれば合計 0）
    assert sum(w[dates[3]].values) == pytest.approx(0.0)
