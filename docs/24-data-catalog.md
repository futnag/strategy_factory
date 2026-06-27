# 24 - データカタログ（`data/` 案内・AIエージェント向け）

このドキュメントは、`data/` 配下に**どこに・どんなデータが・どんな形式で**保存されているか、
そして**どう読むのが正しいか**を一元的に案内する。コーディングエージェント／AIエージェントが
最初に読むべき索引。人間の開発者にも有効。

- **正本**: このファイル（`docs/24-data-catalog.md`）。スキーマや方針の変更時はここを更新する。
- **ローカル索引**: `data/README.md`（中身は本ファイルへのポインタのみ。`data/` を `ls` した
  エージェントが必ず辿れるよう設置。`.gitignore` の `!data/README.md` 例外でコミット対象）。
- プログラム的カタログは `invest_system/data/catalog.py`（差分更新エンジンが参照する正準定義）。

---

## 0. ゴールデンルール（先に読む）

1. **`data/` は丸ごと git 管理外**（`.gitignore` の `/data/` ＝ルートアンカー）。例外は
   `data/README.md` のみ。**市場データ・取得物を絶対にコミットしない**（J-Quants / investing.com /
   M&A Online の各 ToS、および EDINET 取得物の容量）。コミット可能なのは名簿
   （`invest_system/equities/activist_registry.csv`）や様式コード表などの**メタのみ**。
2. **生パスを直読みするより、ローダ経由を優先**。略名解決・空マーカー除外・Silver高速パス・
   PIT整合がローダ側に実装済み（§3 の各「ロード」を参照）。
3. **PIT（ポイントインタイム）規律を死守**。提出日／公表日アンカー、`adj_*` は分割で過去が
   書き換わる派生値、`_displayed` 系（M&A Online）はバンプ後の最終値＝**エントリには使えない**。
   詳細は §5。
4. **銘柄コードは英字混じり5桁文字列**（例 `130A0`, `135A0`）。`int` 化しないこと。
5. **`_empty` 列を持つ parquet は「その日データ無し」のマーカー**（祝日・未開示）。読み飛ばす。

---

## 1. クイックリファレンス

| データセット | パス | 形式・粒度 | ロード | 出所（生成元） | 更新 |
|---|---|---|---|---|---|
| 全銘柄日次OHLCV | `jquants/daily/{YYYYMMDD}.parquet` | parquet・1日1ファイル・全銘柄 | `panel.load_daily_panel()` / `store.load_wide()` | `sources/jquants.py` | 日次・追記 |
| 銘柄別日次（オンデマンド） | `jquants/daily_by_code/{code}.parquet` | parquet・1銘柄1ファイル | `jq.fetch_daily_history(code)` | 〃 | 随時 |
| 財務サマリー（全件） | `jquants/fins_summary/{YYYYMMDD}.parquet` | parquet・1日1ファイル | `fundamentals.load_fundamentals()` | 〃 | 日次・追記 |
| 財務（銘柄別・旧） | `jquants/statements/{code}.parquet` | parquet・1銘柄1ファイル | 〃（自動併合） | 〃 | 随時 |
| 指数四本値 | `jquants/indices/code_{idx}.parquet` | parquet・1指数1ファイル | `jq.fetch_index_bars()` | 〃 | 週次 refresh |
| 投資部門別売買 | `jquants/investor_types/all.parquet` | parquet・固定1本（全履歴） | `jq.fetch_investor_types()` | 〃 | 週次 refresh |
| 空売り残高 | `jquants/short_positions/{calc_YYYYMMDD}.parquet` | parquet・算定日別 | `jq.fetch_short_positions()` | 〃 | 日次・追記 |
| 業種別空売り比率 | `jquants/short_ratio/s33_{sector}.parquet` | parquet・33業種別 | `jq.fetch_short_ratio()` | 〃 | 随時 |
| 信用週末残 | `jquants/margin_weekly/date_{YYYYMMDD}.parquet` | parquet・週末日別 | `jq.fetch_weekly_margin()` | 〃 | 週次・追記 |
| 日々公表信用残 | `jquants/margin_alert/{date}.parquet` | parquet・日別 | `jq.fetch_margin_alert()` | 〃 | 日次・追記 |
| 日経225オプション | `jquants/options_225/{YYYYMMDD}.parquet` | parquet・日別・全契約 | `jq.fetch_index_options()` | 〃 | 日次・追記 |
| **Silver wide パネル** | `processed/equities/wide/{field}.parquet` | parquet・index=Date×col=Code | `store.load_wide()` | `data/store.py` | 派生・materialize |
| Silver long（SQL用） | `processed/equities/long/year={YYYY}/` | parquet・yearパーティション | DuckDB/`read_parquet` | 〃 | 派生・全再構築 |
| **Gold 特徴量** | `features/{name}.parquet` | parquet・index=Date×col=Code（float32） | `feature_store.load_feature()` | `data/feature_store.py` 他 | 派生・再計算 |
| EDINET 書類一覧 | `edinet/list/{YYYYMMDD}.parquet` | parquet・日別メタ（29列） | `ed.fetch_documents_list()` | `sources/edinet.py` | 日次・追記 |
| EDINET 生書類 | `edinet/docs/{docID}_5.zip` | zip（XBRL→CSV type=5） | `ed.fetch_document()` | 〃 | 随時 |
| TOB 案件表（C1） | `edinet/tob_deals.parquet` | parquet・1案件1行（422件） | 直接 read | `equities/tob_events.py` | 随時・再構築 |
| 大量保有報告（C2） | `edinet/large_holdings.parquet` | parquet・1報告1行 | 直接 read | `equities/large_holdings.py` | 随時・再構築 |
| 大量保有 変更報告 | `edinet/large_holdings_changes.parquet` | parquet・1報告1行 | 直接 read | 〃 | 随時・再構築 |
| EDINET 財務 long | `edinet/fundamentals_long.parquet` | parquet・1開示1行（44k） | `edinet_fundamentals.edinet_fundamentals_panel()` | `equities/edinet_fundamentals.py` | 随時・再構築 |
| EDINET-DB 財務 | `edinet/edinetdb/financials/{Ecode}.parquet` | parquet・1社1ファイル | — | `sources/edinetdb.py` | 随時 |
| クロスアセット価格 | `investers/*.parquet`（日本語名） | parquet・OHLCV・日次 | `external.load_external_prices()` | 手動DL→`clean_investing_csv.py` | 手動 |
| マクロ（FRED/yf） | `supplemental/*.parquet` | parquet・index=date | `external.load_macro()` | `data/fetch_supplemental_data.py` | 随時 |
| FF Japan ファクター | `external_factors/ff_japan_*.parquet` | parquet・index=yyyymm | 直接 read | 公式FFデータ由来 | 随時 |
| **研究行列（GKX）** | `phase4/`, `phase4b/` | X/y/universe/macro/meta | 直接 read（§4） | `examples/build_phase4*_matrix.py` | 随時・再構築 |
| 運用成果物（Phase2） | `phase2/` | manifest/orders/fills 月次 | 直接 read | `invest_system/production/` | 月次 |
| 研究HTMLレポート | `reports/*.html` | HTML | ブラウザ | `examples/research_*.py` | 随時 |
| **試行レジストリ** | `research_trials.db` | SQLite（`trials` 表） | `invest_system/validation/` | 検証ハーネス | 随時・追記 |
| M&A Online（TOB履歴） | `manual/maonline/` | md（生）＋csv（parsed） | `parsed/*.csv` を read | 手動DL→`parse_maonline_md.py` | 手動 |
| **TDnet 適時開示一覧** | `tdnet/{YYYYMMDD}.parquet` | parquet・1日1ファイル | `tdnet.load_tdnet()` / `filter_tagged()` | `sources/tdnet.py` | 日次・前向き蓄積 |
| **JPX ToSTNeT 超大口** | `jpx_tostnet/{YYYYMMDD}.parquet` | parquet・1日1ファイル | `jpx_tostnet.load_tostnet()` | `sources/jpx_tostnet.py` | 日次・前向き蓄積 |

---

## 2. アーキテクチャ：4層（メダリオン）

```
L0 Raw  ─ data/jquants/*  data/edinet/{list,docs}     ← API生ミラー（不変・追記/by-code）
   │       sources/jquants.py · sources/edinet.py
   ▼  store.materialize_wide / rebuild_adjusted
L1 Silver ─ data/processed/equities/{wide,long}        ← Date×Code wideパネル
   │       data/store.py
   ▼  feature_store.materialize_* / 各 equities factor
L2 Gold  ─ data/features/*                              ← PIT安全な派生特徴量
   │       data/feature_store.py
   ▼  examples/build_phase4*_matrix.py
L3 研究行列 ─ data/phase4{,b}/{X,y,universe,macro,meta} ← GKX横断ML設計行列
```

- **L0 Raw** は append-only・冪等・再開可能（取得最適）。by-date（その日の全銘柄）と by-code の2系統。
- **L1 Silver** はフィールド別 wide（1ファイル/フィールド）。`AsOfView`／横断分析が直接消費。
  生(無調整)は純追記、`adj_*` は「生 × 累積調整係数」で**全体再構築**（分割が過去を書き換えるため）。
- **L2 Gold** は `adj_close` 等から**因果的に**計算（f[t] は ≤t のみ参照）。特徴量は進化するため
  差分追記でなく**再計算**。
- **L3** は各 phase の事前登録に紐づく一回限りの設計行列（§4）。

EDINET 系（イベント戦略 C1/C2・財務）は別ベース `data/edinet/`（別APIキー）で、上記とは独立。

---

## 3. データセット別 詳細

### 3.1 `jquants/` — J-Quants API ミラー（コア市場データ）

出所はすべて `invest_system/data/sources/jquants.py`。差分更新は `catalog.py` の `DATASETS` /
`REFRESH_DATASETS` を `updater.py`（`examples/` の夜間ジョブ）が回す。レート制限自衛あり
（`J_QUANTS_MIN_INTERVAL` 既定12.5秒＝Free安全側、429は60秒〜バックオフ）。

#### `daily/{YYYYMMDD}.parquet` — 全銘柄日次 OHLCV
- 1日1ファイル・その日の全上場銘柄（例 4,335行）。2016-06〜。列16本。
- 主要列: `Date, Code, O, H, L, C`（始高安終）, `UL, LL`（**ストップ高/安ヒットフラグ＝0/1**。当日に
  制限値幅へ達したか否か。**値幅境界の「価格」ではない**＝`close>=UL` 等の価格比較は誤り。§5-10）,
  `Vo`（出来高）, `Va`（売買代金）, `AdjFactor`（調整係数）, `AdjO/AdjH/AdjL/AdjC`（調整後OHLC）, `AdjVo`。
- **ロード（推奨）**:
  ```python
  from invest_system.equities.panel import load_daily_panel
  adj_close = load_daily_panel("AdjC", start="2020-01-01")   # Date×Code wide（Silver優先で高速）
  ```
  低水準で wide を直接読むなら `store.load_wide("adj_close", start=...)`。

#### `daily_by_code/{code}.parquet` — 銘柄別日次（オンデマンド）
- `daily/` と同スキーマ。研究時に特定銘柄の長期系列が要るとき `jq.fetch_daily_history(code)` が作る。
- 全営業日の全銘柄ミラーは `daily/` 側（by-date）が担う方針（コア網羅＋オンデマンド）。

#### `fins_summary/{YYYYMMDD}.parquet` ＋ `statements/{code}.parquet` — 財務
- `fins_summary/` は全件 by-date（その日の全開示企業）。`statements/` は旧 by-code キャッシュ。
- 107列。主要: `DiscDate`（開示日）, `Code`, `DiscNo`, `DocType`, `Sales`（売上）, `OP`（営業利益）,
  `OdP`（経常利益）, `NP`（純利益）, `EPS, BPS, Eq`（純資産）, `EqAR`（自己資本比率）, `TA`（総資産）,
  `CFO/CFI/CFF, CashEq`, `Div*`（配当各種）, `F*`（会社予想）。
- **ロード**: `fundamentals.load_fundamentals(codes=...)` が両者を併合・重複除去
  （キー `Code, DiscDate, DiscNo`）して長形式（行=開示）で返す。`point_in_time` にそのまま渡せる。

#### `indices/code_{idx}.parquet` — 指数四本値
- 1指数1ファイル（`code_0000`=TOPIX 等）。列 `Date, Code, O, H, L, C`。`jq.fetch_index_bars()`。

#### `investor_types/all.parquet` — 投資部門別売買
- 固定1本（全履歴・56列）。13主体 × `Sell/Buy/Tot/Bal`。主体略号:
  `Prop`自己, `Brk`委託, `Tot`総計, `Ind`個人, `Frgn`海外, `SecCo`証券, `InvTr`投信,
  `BusCo`事業法人, `OthCo`その他法人, `InsCo`生損保, `Bank`都銀地銀, `TrstBnk`信託, `OthFin`その他金融。
- 期間つきの履歴ファイル（`from_..._to_....parquet`）も併存。`jq.fetch_investor_types()`。

#### `short_positions/{calc_YYYYMMDD}.parquet` — 空売り残高（個別）
- 算定日別。`ShrtPosToSO`（発行済比）, `ShrtPosShares`, `ShrtPosUnits`, `PrevRptRatio` 等。

#### `short_ratio/s33_{sector}.parquet` — 業種別空売り比率
- 33業種コード別。`Date, S33, SellExShortVa`（空売り除く売り）, `ShrtWithResVa`（信用残あり）,
  `ShrtNoResVa`（信用残なし）。

#### `margin_weekly/date_{YYYYMMDD}.parquet` / `margin_alert/{date}.parquet` — 信用残
- 週末残（週次）と日々公表残（日次）。**`margin_weekly` は `_empty` マーカーが多い**（非対象日）。

#### `options_225/{YYYYMMDD}.parquet` — 日経225オプション四本値＋IV
- 日別・全契約（例 4,868行・30列）。`O/H/L/C`（日中）, `EO/EH/EL/EC`（イブニング）,
  `AO/AH/AL/AC`（立会）, `OI`（建玉）, `Strike`（権利行使）, `Settle`（清算値）, `Theo`（理論値）,
  `IV`（インプライドボラ）, `UnderPx`（原資産）, `IR`。

### 3.2 `processed/equities/` — Silver（`data/store.py`）

- `wide/{field}.parquet`: index=Date, columns=銘柄コード。1ファイル/フィールド
  （`open, high, low, close, volume, turnover, adj_factor, upper_limit, lower_limit` ＋
  派生 `adj_close, adj_open, adj_high, adj_low`）。規模 約2,444日 × 5,371銘柄。
  ⚠ **`upper_limit`/`lower_limit` は 0/1 のストップ高/安フラグ**（価格ではない）。引け張り付き
  （執行不能）の判定は `equities/frictions.py:limit_lock_flags`＝`UL==1 かつ close>=high` を使う。
  ```python
  from invest_system.data.store import load_wide
  c  = load_wide("close")        # 生終値
  ac = load_wide("AdjC")         # 別名OK（"AdjC"/"C"/"Va" 等→正準名に解決）
  ```
- `long/year={YYYY}/`: DuckDB/SQL・将来のティック/分足スケール用。日次運用では不要。
- 生成: `store.materialize_wide()`（増分・冪等）→ `store.rebuild_adjusted()`（係数変化時のみ実質変化）。
  健全性点検は `store.health_check()`（被覆・NaN率・期間・フィールド間日付整合）。

### 3.3 `features/` — Gold 派生特徴量（`data/feature_store.py` 他）

- 45本。各 wide（Date×Code・float32）。価格由来の普遍特徴（`returns, log_returns, rvol_*,
  mom_*, reversal_*, beta, ivol, max_ret, ret_skew` …）、流動性/マイクロストラクチャ
  （`amihud_illiq, dollar_volume, turnover, zero_ret_days, parkinson_vol, garman_klass_vol,
  corwin_schultz_spread, roll_spread, vpin, rsi` …）、信用/空売り（`margin_*, short_*,
  days_to_cover, sector_short_ratio`）、ファンダ（`sue_*, forecast_revision, *_growth,
  *_stability, sustainable_growth`）、その他（`high_52w, seasonality, dimson_beta, regime`）。
  ```python
  from invest_system.data.feature_store import load_feature
  beta = load_feature("beta", start="2024-01-01")
  ```
- すべて `adj_close` 等から**因果的**に計算（先読みなし）。`regime.parquet` は vol三分位＋トレンド。

### 3.4 `edinet/` — EDINET 開示データ（イベント戦略・財務）

別APIキー（`.env` の `EDINET_API_KEY`）・別ベース。詳細は `docs/08`（API）, `docs/09`（C1）,
`docs/10〜12`（C2）, `docs/14`（財務）。

- `list/{YYYYMMDD}.parquet`: 書類一覧メタ（29列）。`docID, edinetCode, secCode, filerName,
  ordinanceCode`（府令: `040`第三者TOB/`050`自社TOB/`060`大量保有）, `formCode, docTypeCode,
  parentDocID, submitDateTime`（**PITアンカー**）, `periodStart/End` 等。
  ⚠ `formCode` の `01xx=新規/03xx=変更/09xx=訂正` という以前の推定は**誤り**。新規/変更/訂正の
  区別は**本体の `DocumentTitleCoverPage`**でのみ確定する（`large_holdings.report_class()`）。
- `docs/{docID}_5.zip`: 生 XBRL→CSV（type=5・UTF-16 TSV・9列）。約53,544本。`ed.fetch_document()`。
- `tob_deals.parquet`（C1・422件・23列）: `deal_id, announce_dt, acquirer_*, target_*, competing,
  initial_price`（当初価格・**PIT用**）, `final_price`, `purchase_ratio_pct, period_start,
  period_end_initial/final, n_bumps, n_extensions, result`（成否）, `withdrawn, body_ok`。
- `large_holdings.parquet`（C2・4,872件）/ `large_holdings_changes.parquet`（変更・2,903件）:
  `doc_id, submit_dt, issuer_edinet, sec_code, report_class, purpose, is_important_proposal`
  （**C2ユニバース定義＝purpose に「重要提案行為等」**）, `holding_ratio(_prev), shares_held,
  filer_name_jp/en, canonical_group, style, in_registry, body_ok`。
- `fundamentals_long.parquet`（44,478行・25列）: EDINET財務の long。`docID, Code, DiscDate,
  period_end, basis, net_sales, operating_income, ordinary_income, profit, total_assets,
  net_assets, equity, cfo/cfi/cff, cash, rd_expense, shares_outstanding, interest_debt` 等。
  ロード: `edinet_fundamentals.edinet_fundamentals_panel(rebal_dates, fields)`。
- `edinetdb/financials/{Ecode}.parquet`: EDINET-DBコネクタ由来の社別財務（17列）。
  `edinetdb/quota.json` は当日API消費カウンタ。

### 3.5 `investers/`, `supplemental/`, `external_factors/` — クロスアセット・マクロ

- `investers/*.parquet`: **investing.com から手動DL**した OHLCV（FX/指数/先物/商品・2010〜・
  **日本語ファイル名**）。列 `open, high, low, close, volume, change_pct`。ToSによりコミット不可。
- `supplemental/*.parquet`: FRED/yfinance 由来マクロ（`fred_macro, macro_extended, cpi, vix,
  n225_iv, commodities, currencies, us_market, macro_us_combined`）。index=date。
- `external_factors/ff_japan_3factors.parquet`（`Mkt-RF, SMB, HML, RF`）/ `ff_japan_momentum.parquet`。
  index=yyyymm。
- **ロード（英語キーで正準化）**:
  ```python
  from invest_system.data.external import load_external_prices, load_macro, asof_align, list_external
  px  = load_external_prices(["usdjpy", "nk225", "topix"], field="close")  # 日本語名を解決
  mac = load_macro(["japan_10y_yield", "vix", "japan_cpi"])
  # JPリバランス日に ≤t-lag で as-of 結合（先読みなし）
  aligned = asof_align(px, rebal_dates, lag_days=1)
  list_external()   # 利用可能キー一覧
  ```

### 3.6 `phase4/`, `phase4b/` — GKX 横断ML 設計行列 → §4

### 3.7 `phase2/` — 無人運用（Phase2）成果物

月次。`manifest_{YYYY-MM}.json`（資本・判定日・ヘッジ等）, `intended_*.parquet`（意図ポジ）,
`fills_*.parquet`（約定: `key, fill_date, fill_price`）, `orders_{eq,ts}_*.csv`, `equity_daily.csv`,
`status.json`, `report_latest.md`, `months.csv`。生成は `invest_system/production/`。

### 3.8 `reports/` — 研究HTMLレポート

`examples/research_*.py` が吐く可視化（28本＋`index.html`）。戦略仮説の記述統計・図。

### 3.9 `research_trials.db` — 試行レジストリ（DSR/PBO 事前登録台帳）

SQLite・`trials` 表（742行）。列: `trial_id, uuid, scope, strategy_id, hypothesis,
economic_rationale, params_json, status, sharpe, n_obs, skew, kurt, returns_hash, extra_json,
fingerprint, preregistered_at, completed_at`。**多重検定補正（DSR）と事前登録規律の根幹**＝
試行数 K をここで管理する。`invest_system/validation/` 経由で読み書き。

### 3.10 `manual/maonline/` — M&A Online（TOB履歴・主に2022年以前）

- `tob_md/yearly/{year}.md`, `tob_md/detail/{year}/{id}.md`: 生markdown（手動コピー・約1,240本）。
  **スクレイピング禁止**（手動DLのみ。自動巡回不可）。
- `parsed/tob_list.csv`, `parsed/tob_detail.csv`, `parsed/yearly_summary_derived.csv`,
  `validation_report.txt`: `parse_maonline_md.py` による parsed 出力。
- ⚠ **`_displayed` 系列（`price_displayed_jpy, premium_displayed_pct, end_date_displayed`）は
  バンプ後の最終値**。エントリ判断には使えない（PIT非安全）。当初価格は EDINET 側を使う。

### 3.11 `tdnet/`・`jpx_tostnet/` — オルタナティブデータ（前向き蓄積）

公式 API なし・**バックフィル不可**＝今日から蓄積する種まきデータ。詳細設計は `docs/47`。

#### `tdnet/{YYYYMMDD}.parquet` — TDnet 適時開示一覧
- 1日1ファイル。列: `disclosure_date/time, code, company_name, title, pdf_url, xbrl_url,
  exchange, doc_id, event_tags, source`。
- **ロード**:
  ```python
  from invest_system.data.sources.tdnet import load_tdnet, filter_tagged
  df = load_tdnet(start="20260601")
  buybacks = filter_tagged(df, "buyback_announce")
  ```
- 取得: `examples/update_tdnet.py`（日次・冪等）。公開閲覧は**約1ヶ月のみ**。
- `event_tags` は表題キーワード分類（`buyback_announce`, `tob_related`, `guidance_revision` 等）。

#### `jpx_tostnet/{YYYYMMDD}.parquet` — ToSTNeT 超大口約定（≥50億円）
- 1日1ファイル。9列（J-Quants Pro `/prices/tostnet_super_large_lot` と同一構造）。
- **ロード**: `from invest_system.data.sources.jpx_tostnet import load_tostnet, daily_summary`
- 取得: `examples/update_tostnet.py`。**2週間以内ごと**に実行（ページは約2週間で消える）。
- 方針書: `tostnet_monitoring_plan.md`

---

## 4. 研究行列 `phase4/`・`phase4b/`（GKX）

一回限りの設計行列。`phase4b` は **小型株ユニバース**版（`docs/21,22`）。`phase4` は本体（`docs/19,20`）。

| ファイル | 形 | 内容 |
|---|---|---|
| `X.parquet` | MultiIndex `[Date, Code]` × 60列 | 特徴量（`meta.json:char_cols`）。例 170,182行 |
| `y.parquet` | MultiIndex `[Date, Code]` × `y` | 翌月フォワードリターン（先読みなし） |
| `universe.parquet` | Date × Code（bool） | 各月の対象ユニバース |
| `macro.parquet` | Date × 5 | `jp_10y, term_spread, vix, n225_iv, foreign_flow` |
| `meta.json` | — | `months, t1, char_cols(60), macro_cols(5), meta{n_features, delist_policy, lag_days}` |

```python
import pandas as pd, json
X = pd.read_parquet("data/phase4b/X.parquet")        # index=[Date, Code]
y = pd.read_parquet("data/phase4b/y.parquet")
meta = json.load(open("data/phase4b/meta.json", encoding="utf-8"))
```

生成: `examples/build_phase4_matrix.py` / `examples/build_phase4b_matrix.py`。
`meta.delist_policy="last_price"`, `lag_days=1`（PIT規律）。

---

## 5. 地雷集（PIT・規律）

1. **`data/` は非コミット**。市場データ・EDINET取得物・M&A Online md を絶対に git に入れない。
   コミット可は名簿（`activist_registry.csv`）と様式コード表など**メタのみ**。
2. **提出日／公表日アンカー**。EDINET は `submitDateTime`、TOB は公表日 T+1、大量保有は提出日 T+1。
3. **`adj_*` は分割で過去が back-adjust され書き換わる**派生値。生(無調整) `open/high/...` とは別物。
   ヒストリカルな「当時の価格」が要る局面で `adj_*` を使わない。
4. **`_displayed`（M&A Online）はバンプ後最終値＝エントリ非安全**。当初価格は EDINET `initial_price`。
5. **EDINET 閲覧窓は約4〜5年**（古い書類はレコードが null 化）。実効的な TOB/大量保有窓は **2022年以降**。
   それ以前は手動DLの M&A Online を使う。
6. **`_empty` 列 = データ無しマーカー**（祝日・未開示）。ローダは除外済みだが直読み時は要注意。
7. **銘柄コードは英字混じり5桁文字列**（`130A0`）。`int` 化・ゼロ詰め前提のコード禁止。
8. **C2ユニバースは purpose フィルタで定義**（`is_important_proposal`＝「重要提案行為等」）。
   名簿で手選びするのは in-sample 選択＝禁止。名簿はあくまで名寄せ/スタイルタグ/重複排除の補助。
9. **試行数 K は `research_trials.db` で管理**。閾値の後付けチューニング（hindsight）禁止。
10. **`UL`/`LL`（wide では `upper_limit`/`lower_limit`）は 0/1 のストップ高/安ヒットフラグ**＝値幅境界の
    「価格」ではない。ほぼ全行 0 のため `close>=UL` のような価格比較は常時 True 化して破綻する（実害例
    あり）。ストップ高/安**イベント**は `UL==1`/`LL==1`、**引け張り付き**（執行不能）は
    `frictions.limit_lock_flags`（`UL==1 かつ close>=high` / `LL==1 かつ close<=low`）で判定する。

---

## 6. 環境・前提

- Python は `.venv`（純 Python: numpy/pandas/pyarrow/scipy/sklearn/statsmodels）。
- APIキーは `.env` のみ（`J_QUANTS_API_KEY`, `EDINET_API_KEY`）。**コード/git/ログに出さない**。
- ほぼ全データが parquet（pyarrow）。日付は `datetime64`、コードは文字列。
- 読取は基本ネット不要（キャッシュ読取）。取得（fetch_*）のみネットワーク＆レート制限あり。

---

## 7. 新しいデータを足すときの手順（チェックリスト）

1. 取得は `sources/`（API）か手動DL（ToS厳守・スクレイピング禁止）。生は L0 に append-only で置く。
2. `catalog.py` に `Dataset`/`RefreshSpec` を登録（差分更新に乗せる場合）。
3. 横断分析で使うなら Silver（`store.materialize_*`）→必要なら Gold（`feature_store`）へ。
4. **本ファイル（`docs/24`）のクイックリファレンス表とデータセット別詳細に追記**。
5. PIT 影響（アンカー・先読み・調整）を §5 の観点で確認。
6. コミット対象か（メタのみ可・市場データ不可）を確認。`.gitignore` に穴を開けない。
