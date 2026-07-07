---
name: kabu-dd
description: 個別銘柄の定型デューデリジェンス。EDINET DB MCP（財務10年/セグメント/株主・大株主異動/アクティビスト/役員報酬/関連当事者/KPI）＋ローカルミラー（株価・信用/空売り・セクター内バリュエーション分位）＋WebSearch（直近ニュース・進行中の材料の最新確認・必須）で1本のDDレポートを output/kabu/ に生成。裁量レーン＝レジストリ外・投資助言ではなく情報整理。引数: 銘柄コード or 社名 [--peers] [--forensic] [--watch]。
---

# /kabu-dd — 個別銘柄デューデリ（1起動＝1銘柄1レポート）

## ガードレール（違反不可）

1. **裁量レーン**: 出力は `output/kabu/` のみ。research_loop/・BACKLOG・IDEAS・レジストリ・
   docs には一切書かない（研究に載せるなら人間が /research-scout 経由で別途）。
2. **投資助言ではない**: 「買い/売り」の断定をしない。事実整理＋強み/弱み/リスクフラグ＋
   検証すべき論点、まで。
3. **全データに日付を付す**: 財務は DiscDate、株価・需給はデータ日付。EDINET DB は
   外部サービス（Cabocia）であり一次ソースは EDINET/決算短信、と明記する。
4. 発注・watchlist 変更は `--watch` 明示時のみ（add_to_watchlist）。

## 手順

### Step 0 準備
- MCP ツールが deferred なら ToolSearch で**一括ロード**（search_companies, get_company,
  get_financials, get_segments, get_earnings, get_shareholders, get_shareholder_history,
  get_activist_positions, get_directors, get_director_compensation,
  get_related_party_transactions, get_cross_shareholdings, get_kg_company_summary,
  get_kg_kpi_track_record, get_events, get_industry_benchmark, find_peer_strategies,
  add_to_watchlist から必要分）。
- 入力が社名なら search_companies で4桁コードに解決（複数候補は確認）。

### Step 1 収集（EDINET DB — コア）
get_company / get_financials（可能な限り長期）/ get_segments / get_earnings /
get_shareholders + get_shareholder_history / get_activist_positions /
get_kg_company_summary + get_kg_kpi_track_record / get_events。
ガバナンス系（directors・報酬・関連当事者・持合い）は異常の兆候があれば深掘り、
なければ要約1行で済ませる（全ツールを無差別に叩かない）。

### Step 2 収集（ローカルミラー — docs/24 §3.1 参照）
インライン python で:
- 株価・出来高: `data/jquants/daily_by_code/{code}.parquet`（無ければ wide パネル）
  → 1y/3y リターン・ボラ・ADV（売買代金）・52週高値距離。
- 需給: `margin.load_weekly_margin()` を Code で絞り short_to_long 推移、
  `margin.short_interest(load_short_positions())` の直近値。
- バリュエーション分位: `fundamentals.load_fundamentals()` の直近 BPS/EPS × 株価で
  B/M・E/P を計算し、`equities_master.parquet` の同一 S33 内で percentile。
- ToSTNeT/TDnet: `data/jpx_tostnet/`・`data/tdnet/` の直近に当該銘柄があれば記載。

### Step 2.5 Web 最新確認（必須）
EDINET DB・ローカルミラーには更新遅延があるため、**直近の重要事実は Web が正**:
直近1-3ヶ月のニュース・適時開示（会社IR・株探・日経）・株価急変日の理由・進行中の材料
（TOB/資本提携/訴訟/不祥事/月次動向）。重大事実（TOB 進行中・監理銘柄等）が見つかったら
レポート冒頭に明記する。

### Step 3 オプション
- `--peers`: find_peer_strategies / get_industry_benchmark ＋同一 S33 の主要銘柄で
  比較表（成長・利益率・バリュエーション・需給）。
- `--forensic`: 収益の質スクリーン＝ΔDSO・受取債権成長 vs 売上成長・アクルーアル
  （`examples/research_dso_quality.py` の定義を流用）。**注記必須**: これはリターン因子では
  なく blow-up 回避の品質スクリーン（docs/29・dso_quality scope の知見）。
- `--watch`: add_to_watchlist で登録し、登録した旨を報告。

### Step 4 レポート
`output/kabu/{code}_{YYYYMMDD}.md` に固定構成で出力:
①会社概要（1段落） ②事業・セグメント ③財務トレンド（成長・利益率・CF・還元）
④資本政策・ガバナンス（PBR改革対応・資本コスト開示の有無= `data/tse_capital_disclosure/`）
⑤株主構成・需給（大株主異動・アクティビスト・信用/空売り） ⑥バリュエーション（絶対＋S33内分位）
⑦リスクフラグ（forensic 含む） ⑧カタリスト/検証すべき論点 ⑨データ出典と日付一覧。
最後にユーザー向け1画面サマリ（③⑥⑦⑧の要点）。

## 運用メモ
- 深さの既定は「コア収集＋レポート」で1銘柄15分相当。`--peers --forensic` はフル版。
- EDINET DB の欠損（小型株で頻出）はローカル EDINET 系（`edinet/fundamentals_long.parquet`・
  `edinet/edinetdb/financials/`）で補完を試みてから「欠損」と報告する。
