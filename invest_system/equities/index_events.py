"""日経225 構成銘柄変更イベント（公式変更履歴から curate した公開リファレンスデータ）。

出典（検証ずみ）：
- 実施日・銘柄コード・ペア構成：日経インデックス「日経平均株価銘柄変更履歴」(2026/4/1現在)
  https://indexes.nikkei.co.jp/nkave/archives/file/history_of_nikkei_stock_average_component_changes_jp.pdf
  （日付は結合セル＝グループ中央行に印字。グループ構成は当時報道と突合して解決）
- 定期見直しの発表日：当時の日経・Bloomberg 報道で1件ずつ検証（例：2021-09-06
  bloomberg QZ04YAT0AFB601、2023-03-03 nikkei DGXZQOUB0364W0T00C23A3000000、
  2025-03-05 bloomberg SSN2SAT0AFB400）。2023年から定期見直しは年2回（4月・10月）。

J-Quants の市場データ（ToS によりコミット不可）と異なり、これは公開リファレンス情報
なのでリポジトリにコミットする。

PIT 規律：発表日（announce）以前にイベントを知ることはできない。announce=None の臨時
イベントは発表日が未検証＝発表日アンカーの戦略（ドリフト）から構造的に脱落する
（`event_holding_windows` が落とす）。実施日のみ使う戦略（実施後リバーサル）には使える。

kind（フロー仮説への適合性）：
  periodic      … 定期見直しの裁量採用/除外。パッシブ・フローが最大。発表日検証済み
  extraordinary … 臨時補充（既存銘柄の退出に伴う委員会の裁量採用）。発表日 None
  succession    … 統合・持株会社化の継承上場（保有者が引き継ぐ＝新規パッシブ需要なし）
  delisting     … TOB 等で消滅する除外側（価格が TOB 価格に張り付く）
  demotion      … 市場区分降格・規則除外（上場は継続＝取引可能な除外側）
取引可能なレグは `tradeable_event_legs`：採用側 = periodic/extraordinary、
除外側 = periodic/demotion（succession/delisting は除外）。
"""
from __future__ import annotations

import pandas as pd

# (announce, effective, out_code, out_kind, in_code, in_kind, note)
_RECORDS = [
    (None, "2016-08-01", "6753", "demotion", "7272", "extraordinary",
     "シャープ東証2部降格→ヤマハ発動機"),
    (None, "2016-08-29", "8270", "succession", "8028", "succession",
     "ユニーG→ファミマ統合（継承）"),
    ("2016-09-06", "2016-10-03", "4041", "periodic", "4755", "periodic",
     "定期2016：日本曹達→楽天"),
    (None, "2017-01-24", "6767", "delisting", "4578", "extraordinary",
     "ミツミ電機（ミネベア統合）→大塚HD"),
    (None, "2017-08-01", "6502", "demotion", "6724", "extraordinary",
     "東芝2部降格→セイコーエプソン"),
    ("2017-09-05", "2017-10-02", "3865", "periodic", "6098", "periodic",
     "定期2017：北越紀州製紙→リクルートHD"),
    ("2017-09-05", "2017-10-02", "6508", "periodic", "6178", "periodic",
     "定期2017：明電舎→日本郵政"),
    ("2018-09-05", "2018-10-01", "5715", "periodic", "4751", "periodic",
     "定期2018：古河機械金属→サイバーエージェント"),
    (None, "2018-12-26", "5413", "delisting", "4631", "extraordinary",
     "日新製鋼（完全子会社化）→DIC"),
    (None, "2019-03-18", "6773", "delisting", "6645", "extraordinary",
     "パイオニア上場廃止→オムロン"),
    (None, "2019-03-27", "5002", "delisting", "5019", "succession",
     "昭和シェル→出光興産（吸収側の継承）"),
    (None, "2019-08-01", "6366", "demotion", "7832", "extraordinary",
     "千代田化工2部降格→バンダイナムコHD"),
    ("2019-09-04", "2019-10-01", "9681", "periodic", "2413", "periodic",
     "定期2019：東京ドーム→エムスリー"),
    (None, "2020-07-29", "8729", "delisting", "8697", "extraordinary",
     "ソニーFH（TOB）→JPX"),
    ("2020-09-01", "2020-10-01", "4272", "periodic", "9434", "periodic",
     "定期2020：日本化薬→ソフトバンク"),
    (None, "2020-10-29", "8028", "delisting", "3659", "extraordinary",
     "ファミマ（TOB）→ネクソン"),
    (None, "2020-12-02", "9437", "delisting", "6753", "extraordinary",
     "NTTドコモ（TOB）→シャープ"),
    ("2021-09-06", "2021-10-01", "3105", "periodic", "6861", "periodic",
     "定期2021：日清紡HD→キーエンス"),
    ("2021-09-06", "2021-10-01", "5901", "periodic", "6981", "periodic",
     "定期2021：東洋製罐GHD→村田製作所"),
    ("2021-09-06", "2021-10-01", "9412", "periodic", "7974", "periodic",
     "定期2021：スカパーJSAT→任天堂"),
    (None, "2021-12-29", "9062", "succession", None, None,
     "日本通運→NXHD移行（除外側・継承）"),
    (None, "2022-01-05", None, None, "9147", "succession",
     "NIPPON EXPRESS HD（継承上場）"),
    (None, "2022-04-04", "8303", "demotion", "8591", "extraordinary",
     "新生銀行スタンダード移行→オリックス"),
    (None, "2022-09-29", "8355", "delisting", "6594", "extraordinary",
     "静岡銀行（持株会社化）→日本電産"),
    ("2022-09-05", "2022-10-03", "3103", "periodic", "6273", "periodic",
     "定期2022：ユニチカ→SMC"),
    ("2022-09-05", "2022-10-03", "6703", "periodic", "7741", "periodic",
     "定期2022：OKI→HOYA"),
    ("2022-09-05", "2022-10-04", "1333", "periodic", "5831", "succession",
     "定期2022：マルハニチロ→しずおかFG（採用側は継承）"),
    ("2023-03-03", "2023-04-03", "3101", "periodic", "4661", "periodic",
     "定期2023春：東洋紡→オリエンタルランド"),
    ("2023-03-03", "2023-04-03", "5703", "periodic", "6723", "periodic",
     "定期2023春：日本軽金属HD→ルネサス"),
    ("2023-03-03", "2023-04-03", "5707", "periodic", "9201", "periodic",
     "定期2023春：東邦亜鉛→日本航空"),
    ("2023-09-04", "2023-10-02", "5202", "periodic", "4385", "periodic",
     "定期2023秋：日本板硝子→メルカリ"),
    ("2023-09-04", "2023-10-02", "7003", "periodic", "6920", "periodic",
     "定期2023秋：三井E&S→レーザーテック"),
    ("2023-09-04", "2023-10-02", "8628", "periodic", "9843", "periodic",
     "定期2023秋：松井証券→ニトリHD"),
    ("2024-03-04", "2024-04-01", "2531", "periodic", "3092", "periodic",
     "定期2024春：宝HD→ZOZO"),
    ("2024-03-04", "2024-04-01", "5232", "periodic", "6146", "periodic",
     "定期2024春：住友大阪セメント→ディスコ"),
    ("2024-03-04", "2024-04-01", "5541", "periodic", "6526", "periodic",
     "定期2024春：大平洋金属→ソシオネクスト"),
    ("2024-09-04", "2024-10-01", "3863", "periodic", "4307", "periodic",
     "定期2024秋：日本製紙→野村総合研究所"),
    ("2024-09-04", "2024-10-01", "4631", "periodic", "7453", "periodic",
     "定期2024秋：DIC→良品計画"),
    ("2025-03-05", "2025-04-01", "9301", "periodic", "6532", "periodic",
     "定期2025春：三菱倉庫→ベイカレント"),
    (None, "2025-07-04", "9613", "delisting", "6963", "extraordinary",
     "NTTデータG（TOB）→ローム"),
    ("2025-09-08", "2025-10-01", "7762", "periodic", "3697", "periodic",
     "定期2025秋：シチズン時計→SHIFT"),
    (None, "2025-11-05", "6594", "demotion", "4062", "extraordinary",
     "ニデック（特別注意銘柄指定）→イビデン"),
    ("2026-03-05", "2026-04-01", "6674", "periodic", "285A", "periodic",
     "定期2026春：GSユアサ→キオクシアHD"),
    ("2026-03-05", "2026-04-01", "6952", "periodic", "7532", "periodic",
     "定期2026春：カシオ→パンパシHD"),
    (None, "2026-04-01", "7205", "delisting", "543A", "succession",
     "日野自動車→ARCHION（統合持株会社の継承）"),
]


def jq_code(code: str) -> str:
    """4桁/英数の取引所コード → J-Quants 5桁コード（末尾に 0 を付す）。"""
    return f"{code}0"


def n225_changes() -> pd.DataFrame:
    """全変更レコード（1行=公式履歴の1ペア）。announce/effective は Timestamp。"""
    df = pd.DataFrame(_RECORDS, columns=["announce", "effective", "out_code",
                                         "out_kind", "in_code", "in_kind", "note"])
    df["announce"] = pd.to_datetime(df["announce"])
    df["effective"] = pd.to_datetime(df["effective"])
    return df


def tradeable_event_legs(df: pd.DataFrame | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """フロー仮説で取引可能な (adds, dels) レグを返す（コードは J-Quants 5桁）。

    adds: in_kind ∈ {periodic, extraordinary}＝委員会の裁量採用（新規パッシブ買い需要）。
    dels: out_kind ∈ {periodic, demotion}＝上場継続の除外（パッシブ売り）。succession
    （継承＝実質需要なし）と delisting（TOB 張り付き＝取引仮説が成立しない）は落とす。
    列: announce, effective, code, kind。
    """
    df = n225_changes() if df is None else df
    adds = df[df["in_kind"].isin(["periodic", "extraordinary"])]
    adds = pd.DataFrame({"announce": adds["announce"].values,
                         "effective": adds["effective"].values,
                         "code": [jq_code(c) for c in adds["in_code"]],
                         "kind": adds["in_kind"].values})
    dels = df[df["out_kind"].isin(["periodic", "demotion"])]
    dels = pd.DataFrame({"announce": dels["announce"].values,
                         "effective": dels["effective"].values,
                         "code": [jq_code(c) for c in dels["out_code"]],
                         "kind": dels["out_kind"].values})
    return adds, dels


def _locate(dates: pd.DatetimeIndex, anchor: pd.Timestamp, offset: int) -> int:
    """anchor 以降の最初の営業日位置 + offset（dates は昇順前提）。"""
    return int(dates.searchsorted(anchor, side="left")) + offset


def event_holding_windows(legs: pd.DataFrame, dates: pd.DatetimeIndex, *,
                          start_anchor: str = "announce", start_offset: int = 0,
                          end_anchor: str = "effective", end_offset: int = -3
                          ) -> pd.DataFrame:
    """各レグの保有ウィンドウを決定日 index 位置で解決する（PIT）。

    anchor が dates に無い日（休日・発表日が非営業日）は直後の営業日に丸める。
    start_anchor の日付が NaT のレグ（announce 未検証の臨時等）は脱落＝発表日を
    使う戦略には構造的に乗らない。start > end や範囲外は脱落。
    Returns: [code, start, end]（start/end は決定日 Timestamp）。
    """
    rows = []
    for _, r in legs.iterrows():
        a0, a1 = r[start_anchor], r[end_anchor]
        if pd.isna(a0) or pd.isna(a1):
            continue
        i0 = _locate(dates, a0, start_offset)
        i1 = _locate(dates, a1, end_offset)
        if i0 < 0 or i1 >= len(dates) or i0 > i1:
            continue
        rows.append((r["code"], dates[i0], dates[i1]))
    return pd.DataFrame(rows, columns=["code", "start", "end"])


def window_weights(longs: pd.DataFrame, shorts: pd.DataFrame,
                   dates: pd.DatetimeIndex) -> dict:
    """{決定日: ウェイト Series}。各日アクティブなレグを脚内等加重（+1/n, −1/n）。

    longs/shorts は event_holding_windows の出力。両脚が同日にアクティブなら
    ロング合計 +1・ショート合計 −1（ダラーニュートラル）。片脚のみの日はその脚のみ。
    同一銘柄が重複ウィンドウでアクティブでも1票（等加重の二重計上はしない）。
    """
    active: dict[pd.Timestamp, dict[str, set]] = {}
    for side, df in (("L", longs), ("S", shorts)):
        for _, r in df.iterrows():
            for t in dates[(dates >= r["start"]) & (dates <= r["end"])]:
                active.setdefault(t, {"L": set(), "S": set()})[side].add(r["code"])
    out: dict[pd.Timestamp, pd.Series] = {}
    for t, sides in active.items():
        w: dict[str, float] = {}
        if sides["L"]:
            for c in sides["L"]:
                w[c] = w.get(c, 0.0) + 1.0 / len(sides["L"])
        if sides["S"]:
            for c in sides["S"]:
                w[c] = w.get(c, 0.0) - 1.0 / len(sides["S"])
        s = pd.Series(w, dtype="float64")
        out[t] = s[s != 0.0]
    return out
