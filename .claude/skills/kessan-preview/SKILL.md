---
name: kessan-preview
description: 個別銘柄の決算プレビュー（Webから決算予想収集＋ローカル定量）。ローカル＝会社予想（FSales/FOP/FNP/FEPS）・修正履歴・四半期進捗率・過去サプライズと発表翌日反応・次回発表日推定。Web＝アナリストコンセンサス（Kabutan/IFIS/Yahoo等・出典必須）・発表日確定。裁量レーン＝PEADスリーブとは絶縁・研究入力禁止。決算発表後に呼ばれたらレビューモード（実績vs予想と反応の解釈）。引数: 銘柄コード or 社名。
---

# /kessan-preview — 決算プレビュー（1起動＝1銘柄1プレビュー）

## ガードレール（違反不可）

1. **PEAD スリーブと絶縁**: Phase 2 の PEAD はルールで動く。このプレビューを根拠に
   PEAD 建玉への介入・先回りを提案しない（裁量で触りたければ /tactical-brief レーンBの
   3点セット付きで、と案内）。
2. **Web コンセンサスは PIT 不能**＝研究・事前登録・レジストリの入力に使わない（定型注記）。
3. **断定禁止**: 「上振れする」ではなく「◯◯なら上振れ寄り」の条件文で書く。
   会社予想の保守バイアス（guidance_bias は検証済み scope）等、既知の統計は出典付きで使う。
4. 全数値に出典と日付（DiscDate / 取得URL / データ日付）。書き込みは `output/kabu/` のみ。

## 手順

### Step 1 コード解決と発表日
社名なら `data/jquants/equities_master.parquet` か EDINET DB search_companies で4桁コードへ。
次回発表日: Web で確定（「<社名> 決算発表 日程」・会社IRページ）。取れなければ
`equities/events.py` の `days_to_next_announcement`（DiscDate 四半期パターン推定）で近似し
「推定」と明記。**発表が直近過去なら Step 4 のレビューモードに切替**。

### Step 2 ローカル定量（インライン python・docs/24 §3.1）
`fundamentals.load_fundamentals()` を当該コードで絞り:
1. **会社予想**: 最新の FSales/FOP/FNP/FEPS と、期中の修正履歴
   （`equities/events.py` の forecast_revision を流用）。
2. **進捗率**: 直近四半期累計（CurPerType）÷ 通期会社予想（Sales/OP/NP 各段階）。
   同社の過去2-3年の同時点進捗率と並べる（季節性を無視した進捗率は誤読の元）。
3. **過去サプライズと反応**: 過去8回の実績 vs 直前会社予想（earnings_surprise 流用）と、
   発表翌営業日リターン（`data/jquants/daily_by_code/{code}.parquet`）。
   ビート率・平均反応・「ビートでも売られた」回数。
4. 需給の前提: 信用 short_to_long・空売り残高の直近（margin モジュール）＝
   ポジションの偏りは反応の非対称性に効く。

### Step 3 Web コンセンサス（WebSearch / WebFetch・出典URL必須）
- アナリストコンセンサス（営業利益・純利益・目標株価レンジ）: Kabutan 銘柄ページ・
  IFIS コンセンサス・Yahoo!ファイナンス・株予報の順で試す。取れない小型株は
  「コンセンサス無し＝会社予想と進捗のみで評価」と明記して成立させる。
- 直近の月次データ・業界統計・同業他社の先行決算（あれば）。

### Step 4 出力
`output/kabu/kessan_{code}_{YYYYMMDD}.md`:
①発表日（確定/推定） ②予想テーブル（前年同期実績・会社予想・コンセンサス・進捗率）
③論点（上振れ条件 / 下振れ条件 / ガイダンス改定の焦点 / 需給の偏り）
④過去の決算反応統計 ⑤定型注記（ガードレール1・2）。
ユーザーへは②③の要点を1画面で。
**レビューモード**（発表後）: 実績 vs 会社予想 vs コンセンサスの差分・当日/翌日反応・
説明会での修正点を同じ構成で。

## 運用メモ
- /kabu-dd と併用する時は kabu-dd → kessan-preview の順（事業理解が先）。
- watchlist 銘柄の決算週は /news-scan のカレンダーで検知→本スキルで個別深掘り、が動線。
