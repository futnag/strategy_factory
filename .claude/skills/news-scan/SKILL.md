---
name: news-scan
description: 市場に影響しそうなニュース/イベントを捜索し、稼働スリーブ（value↔PEAD switch・TSMOM）と watchlist 銘柄への影響マッピングを作る。WebSearch＋ローカル TDnet/ToSTNeT＋EDINET DB イベント・決算カレンダー。ニュースは PIT 履歴が無いため研究・事前登録の入力に使うことを禁止。引数: テーマ or 銘柄コード（省略時=市場全般）。
---

# /news-scan — 市場ニュース捜索（1起動＝1スキャン1マップ）

## ガードレール（違反不可）

1. **研究入力への使用禁止**: ニュース由来の観察を research_loop/・事前登録・レジストリの
   設計根拠にしない（ニュースはバックフィル不可＝PIT 検証不能）。出力にもこの注記を定型で入れる。
2. **ソース必須**: 各項目に URL または ローカルデータの出典（tdnet/日付）を付す。
   一次ソースで確認できない伝聞は「未確認」ラベル。
3. 提案止まり（発注・ポジション変更の指図をしない）。書き込みは `output/news/` のみ。

## 手順

### Step 1 対象の確定
引数からテーマ（例: 日銀・半導体・PBR改革）or 銘柄を確定。省略時は「市場全般スキャン」。
watchlist は EDINET DB get_watchlist（ToolSearch でロード）で取得。
Phase 2 の現在の建玉は `data/phase2/intended_*.parquet`（最新月）を読み取り専用で参照。

### Step 2 収集（3系統を必ず全部）
1. **Web**: WebSearch で直近1週間中心に。既定クエリ群＝
   「日銀 金融政策/植田」「FOMC/米金利」「東証 市場区分/PBR改革」「TOB 公開買付 発表」
   「自社株買い 大型」「日経平均 指数 入替」「空売り規制/大量保有」＋テーマ/銘柄固有。
2. **ローカル TDnet/ToSTNeT**（前向き蓄積・docs/24 §3.11）: `tdnet.load_tdnet()` の直近
   営業日を filter_tagged で TOB/自社株買い/業績修正/増減配に絞る。`jpx_tostnet.load_tostnet()`
   の直近超大口クロスも列挙（誰の持分移動か Web で補足）。
3. **カレンダー**: EDINET DB get_earnings_calendar / get_events で今後2週間の決算・イベント。
   watchlist・建玉銘柄の該当を強調。

### Step 3 影響マッピング
表形式: | イベント | 日付 | 影響先（市場全体 / value / PEAD / TSMOM / 銘柄コード） |
方向と強さ（↑↑/↑/中立/↓/↓↓ ＋高中低） | 根拠・ソース |
- スリーブへの対応付けの目安: 金利・銀行株関連→value、決算・修正→PEAD、
  トレンド転換級のマクロ→TSMOM、個別は watchlist/建玉のみ。
- 重要度「高」は最大5件（全部重要は無情報と同じ）。

### Step 4 出力
`output/news/scan_{YYYYMMDD}.md` に保存し、ユーザーには
「高重要度5件＋今後2週間のカレンダー要点＋定型注記（ガードレール1）」を1画面で。

## 運用メモ
- 毎朝の自動化は /schedule で登録可能（人間が設定）。
- ToSTNeT 超大口は本 repo 固有のエッジ（無料では他に出回らない）＝毎回必ず見る。
- /market-brief と同日に回す場合は news-scan を後にする（ブリーフの数値状態を文脈に使える）。
