# ToSTNeT 大口取引監視システム 構築方針

**作成日**: 2026-06-26
**改訂**: 2026-06-26（実ページ検証・方針確定を反映。旧版の前提のうち複数が誤りと判明したため大幅改訂）
**目的**: JPX無料ページ「ToSTNeT超大口約定情報」を活用した**監視PoC**。前向きにデータを蓄積し、将来のリサーチ・シグナル化／J-Quants Pro 移行に備える。

---

## 0. 方針確定と検証結果（2026-06-26）

### 0.1 確定した方針（ユーザー決定）
- **目的＝当面は「前向き監視＋データ蓄積」**。本来の狙いはリサーチ・シグナル（既存の検証ファクトリ `judge_grid` で edge を裁く）だが、**無料スクレイプには履歴が無く（ページは直近2週間で消える）即時のバックテストは原理的に不可**。よって今は種まき＝前向き蓄積に徹し、履歴が貯まるか J-Quants Pro 契約時に研究へ着手する。
- **置き場所＝既存リポジトリ `strategy_factory` に統合**。Parquet（`data/`）＋差分更新エンジン（`catalog/updater`）＋毎晩の GitHub Actions ＋既存セクターマスタ(S33)を再利用する。**新規の PostgreSQL/TimescaleDB/AWS は作らない**（データ量が小さく過剰。既存の無人パイプライン／Supabase／Vercel と二重になる）。
- **データ源＝当面は JPX 無料スクレイプ**。本来の最良ソースは J-Quants Pro の API（§1.2）だが法人専用・料金非公開のため保留。

### 0.2 実ページ検証で判明した事実（旧版 §2/§3/§4 の前提を修正）
| 旧版の前提 | 検証結果（実機確認済み） | 対応 |
|---|---|---|
| JS依存の可能性が高い→**Playwright推奨** | **静的HTML**（生 `fetch()` のレスポンスに既にテーブルが入っている） | `requests`＋パーサで十分。**Playwright不要** |
| 「取引内容」のテキストを正規表現で解析 | テキストではなく**日次 Excel(.xlsx) へのリンク** | `.xlsx` をDLし `read_excel` で構造化取得（堅牢） |
| `trade_date` は推定 | ファイル名 `YYYYMMDD_ToSTNeT_Trading_Information.xlsx` の `YYYYMMDD`＝取引日。Excel内にも取引日列 | 推定不要・確定値 |
| データは非常にスパース（0件の日が多い） | **実際は1日数件あり得る**（例: 2026/06/24 取引分は6件以上） | 「0件前提」を緩和。蓄積は想定より速い |
| 過去分も取得できる想定 | Excelは**非連番のCMSフォルダID**配下（例 `t13vrt000001ib45-att`）＝URL推測不可 | **バックフィル不可＝前向き蓄積のみ**（方針と整合） |

### 0.3 Excel の実スキーマ（9列・日英併記）＝ Pro API と一致
`公表日/Publication_Date`・`取引日/Trading_Date`・`約定時刻/Trade_Time`・`銘柄コード/Code`・`銘柄名_日本語/Issue_Name_Japanese`・`銘柄名_英語/Issue_Name_English`・`価格_円/Price_yen`・`売買高_株/Trading_Volume_shares`・`売買代金_円/Trading_Value_yen`

- 数値列はカンマ区切りの**文字列**（例 `5,970,051,048`）→ パース時にカンマ除去・型変換。価格は小数4桁。代金は**円単位**（億円ではない）。
- 実例（2026/06/24取引）: 6861 キーエンス ¥5.97B / 8411 みずほ ¥7.72B / 6857 アドバンテスト ¥7.8B / 7011 三菱重工 ¥5.99B / 285A キオクシアHD ¥6.475B / 5411 JFE …
- この9列は J-Quants Pro `/prices/tostnet_super_large_lot`（`PublicationDate/Date/TradeTime/Code/CompanyName/CompanyNameEnglish/Price/Volume/TurnoverValue`）と**同一構造**。→ ローカルを Pro 準拠で設計すれば、将来 Pro 契約時にスクレイプ分と API 分を同一テーブルに継ぎ目なく合流できる。

---

## 1. 全体方針

### 1.1 フェーズ
- **Phase 1（現在・無料）**: JPX「ToSTNeT超大口約定情報」ページ＋日次Excelをスクレイプ → 既存リポに前向き蓄積＋監視。
- **Phase 2（将来・有料）**: J-Quants Pro の ToSTNeT 系 API へ移行（§1.2）。スキーマ互換なので移行は容易。
- データの性質上、**リアルタイム監視は不可能**（売買代金50億円以上の単一銘柄ToSTNeT-1取引を翌営業日に公表）。日次バッチで十分。

### 1.2 J-Quants Pro（Phase 2 の本命ソース。要・料金確認）
- 個人向け J-Quants API（`api.jquants.com`・`x-api-key`・Free/Light/Standard/Premium。現在 Standard 契約）とは**別商品**の、法人向け **J-Quants Pro**（`api.jquants-pro.com`・**Bearer**）。
- 関連エンドポイント（スクレイプ不要・履歴あり）:
  - `/prices/tostnet_super_large_lot` … ToSTNeT超大口（≥50億）。本PoCがスクレイプする対象そのもの。
  - `/markets/off_auction_distribution` … 立会外分売。
  - `/markets/off_auction_share_buyback` … 自己株式立会外買付（ToSTNeT-3）。
  - （注）`/markets/breakdown`（売買内訳）は**立会内のみ**でToSTNeTは含まない。
- **料金非公開・法人専用**。要問い合わせ: `j-quants@jpx.co.jp`（JPX総研フロンティア戦略部）。個人/小規模での契約可否・履歴期間も同時に確認する。

### 1.3 目的
- 機関投資家・大口の ToSTNeT-1 超大口約定の動向を前向きに把握・記録する。
- セクター別フロー集計・異常検知の素地を作る（既存セクターマスタを活用）。
- 将来、十分な履歴が貯まるか Pro 契約した時点で、既存 `judge_grid` で edge を厳密検証する。

---

## 2. データソースと取得方式（検証済み）

### 2.1 対象
- **URL**: https://www.jpx.co.jp/markets/equities/tostnet/index.html
- 単一銘柄（ToSTNeT-1）で売買代金50億円以上の取引（顧客委託分を除く）を翌営業日に掲載。**過去2週間分のみ**。
- ページ本体: 静的HTMLの表（列: `公表日` | `取引内容`）。`取引内容` セルは日次Excelへの `<a>` リンク（アイコン表示のため画面上はテキストが見えない）。

### 2.2 取得フロー
1. index ページを `requests` で取得（静的HTML。Playwright不要）。
2. 表をパースし、各行から `(公表日, 取引日, xlsx_url)` を抽出（取引日はファイル名 `YYYYMMDD` から）。
3. 未取得の取引日について `.xlsx` をDL → `pandas.read_excel`（要 `openpyxl`）で9列を構造化。
4. 数値列のカンマ除去・型変換、列名を Pro 準拠の正規名へマップ、`source='jpx_scrape'` を付与。
5. その取引日に該当取引が無い場合は「0件」マーカーとして記録（取得済みであることを冪等に残す）。

### 2.3 マナー・規約
- robots.txt は要再確認（WebFetch では 403。実ブラウザでは閲覧可）。アクセスは低頻度（日次1回）＋適切な User-Agent ＋インターバルで礼儀正しく。
- JPX市場データの**再配布は規約上不可**。取得データは `data/`（gitignore）にのみ保持し、リポジトリ／GitHub にコミットしない（既存 J-Quants ミラーと同じ運用）。

---

## 3. データモデル（既存リポ・Parquet）

> 旧版の PostgreSQL+TimescaleDB 案は破棄。データ量が小さく（日数件）、既存スタック（Parquet＋差分更新）で十分。論理スキーマは下記、物理は `data/jpx_tostnet/` 配下の Parquet。

### 3.1 Raw/Clean（取引明細・Pro準拠スキーマ）
| 正規列名 | 由来（Excel / Pro API） | 型 |
|---|---|---|
| `PublicationDate` | 公表日 / PublicationDate | date |
| `Date` | 取引日 / Date | date |
| `TradeTime` | 約定時刻 / TradeTime | time/str |
| `Code` | 銘柄コード / Code | str |
| `CompanyName` | 銘柄名_日本語 / CompanyName | str |
| `CompanyNameEnglish` | 銘柄名_英語 / CompanyNameEnglish | str |
| `Price` | 価格_円 / Price | float |
| `Volume` | 売買高_株 / Volume | float |
| `TurnoverValue` | 売買代金_円 / TurnoverValue | float（円） |
| `source` | — | str（`jpx_scrape` / 将来 `jquants_pro`） |

### 3.2 集計（オンデマンド計算でよい。事前テーブル化は任意）
- 日次×銘柄、日次×市場全体、日次×セクター（既存 S33 マスタで `Code→業種`）。
- `TurnoverValue` から億円換算は集計時に派生。`is_large_day` 等のフラグも集計時に算出。

---

## 4. 技術スタック（既存リポに準拠）
| レイヤー | 技術 | 備考 |
|---|---|---|
| スクレイピング | Python + `requests` ＋ HTMLパーサ | 静的HTML。**Playwright不要** |
| Excel解析 | `pandas.read_excel`（+ `openpyxl`） | 依存を1つ追加 |
| データ処理 | pandas（既存に合わせる） | Polarsは導入しない |
| 保存 | Parquet（`data/jpx_tostnet/`、gitignore） | 既存 `catalog/updater` で差分更新・冪等再開 |
| スケジューリング | 既存 GitHub Actions（毎晩21:30 JST、`strategy-factory-ops`） | 1ステップ追加。AWSは不使用 |
| 可視化（将来） | 既存 Supabase / Vercel ダッシュボード | 「直近の超大口」モジュール追加 |

---

## 5. 実装ステップ（状況：2026-06-26）
1. ✅ **スクレイパ本体** `invest_system/data/sources/jpx_tostnet.py` … 純関数 `parse_index_links` /
   `parse_trading_excel` ＋薄い `fetch_index`/`fetch_excel` ＋冪等 `update_tostnet`/`load_tostnet`/
   `daily_summary`。`tests/test_jpx_tostnet.py`（オフライン・合成xlsx・JPX実ファイル不使用）。
   **ライブ end-to-end 検証済み**（urllib＋UAで403回避＝Playwright不要を実証、実Excelを正準9列でパース）。
2. ✅ **前向き蓄積ランナー** `examples/update_tostnet.py` … `data/jpx_tostnet/{YYYYMMDD}.parquet` に冪等追記。
   **現在窓10営業日＝55件をシード済み**（6/19=16件¥1,850億 等）。※ToSTNeTは2週間ローリングのため
   汎用の by-date `DataUpdater`（任意日付取得前提）ではなく EDINET 同様の専用軽量アップデータにした。
3. ✅ **セクター付与・集計** `sector_map`/`add_sector`/`daily_sector_summary` … 既存 J-Quants マスタの
   S33業種名を**先頭4桁 Code 照合**で付与（実データで未収載0件を確認）。日次×業種の合計代金/件数/銘柄数。
4. ⬜ **無人化（要・決定）**: 日次実行。JPXは**データセンターIPを403する可能性**が高く、クラウド(Actions)は
   要到達性検証。当面はローカル（Windowsタスクスケジューラ・自宅IPで成功実績）が無難。2週間窓は猶予あり。
5. ⬜ **将来**: Pro 料金確認 → 妥当なら `/prices/tostnet_super_large_lot` へ切替（同一スキーマ）＋履歴取得 →
   `judge_grid` で edge 検証。

---

## 6. リスク・注意点
- **履歴ゼロ**: バックフィル不可。リサーチは履歴が貯まるか Pro 契約まで保留（方針として受容済み）。
- **ページ構造変更**: JPXのレイアウト変更に弱い（Excelリンク方式・列名が変わり得る）。パーサは列名で位置を解決し、変化を検知したら記録。
- **規約**: 再配布不可・低頻度アクセス厳守。
- **Pro移行**: 料金非公開・法人専用がボトルネック。スキーマ互換なので技術的移行は容易。

---

## 7. 未来のタスク（TODO）

> **当面の運用＝手動**：`examples/update_tostnet.py` を **2週間以内ごと**（理想は数日に1回）に実行して
> 前向きに蓄積する。ページは2週間ローリングなので、これを超えて空けるとその間は恒久欠測になる。

- [ ] **T1: 日次自動化**（優先・方式未定）。当面は手動運用、追って自動化する。
  - 候補A（推奨）: ローカル **Windows タスクスケジューラ**で毎営業日 `examples\update_tostnet.py`
    （自宅IPで成功実績・無料・確実。PCスリープ時は次回 catch-up。2週間窓で猶予あり）。登録は設定変更のため要確認。
  - 候補B: クラウド **GitHub Actions**（既存夜間基盤 `strategy-factory-ops` 相乗り）。ただし JPX が
    データセンターIPを **403 で遮断する懸念** → **使い捨てワークフローで到達性を1回検証**してから採否を決める。
  - 関連（任意）: 取得後に Supabase へ push → 既存 Vercel ダッシュボードに「直近の超大口」を表示。
- [ ] **T2: J-Quants Pro 確認・移行**。`j-quants@jpx.co.jp` に料金・履歴期間・小規模契約可否を照会 →
  妥当なら `/prices/tostnet_super_large_lot`（同一9列スキーマ）＋`off_auction_distribution`／
  `off_auction_share_buyback` を取得し、スクレイプ分と同一テーブルに合流。
- [ ] **T3: リサーチ着手**（本来目的）。履歴が十分貯まる or Pro 契約後に、既存 `judge_grid`
  （事前登録＋大域デフレートDSR＋PIT＋執行ラグ/コスト/容量）で「ToSTNeT超大口フローに edge があるか」を裁く。
- [ ] **T4: 監視UX**（任意）。`daily_summary` / `daily_sector_summary` のダッシュボード化・閾値アラート。

---

**このドキュメントは随時更新する。**
