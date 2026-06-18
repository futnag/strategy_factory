# 14. EDINET 三表 PIT ファンダメンタルズ（GKX Phase 2）

López de Prado 由来の検証ファクトリを GKX（Gu, Kelly & Xiu 2020）型の機械学習×クロス
セクション予測へ育てる **Phase 2**。金融庁 公式 EDINET API v2 の有価証券報告書 type=5
（XBRL→CSV 変換済み）を一次ソースに、財務三表を **提出日アンカーの Point-in-Time パネル**
として統合し、J-Quants Standard だけでは作れなかった CF 系・BS 明細系の特徴量を解禁する。

> **位置づけ**：これは将来の ML 推定器（Phase 4）の入力**データ基盤**であり戦略探索ではない。
> 永続レジストリ（K）は消費せず判定もしない。予測力の確認は throwaway 診断（IC・被覆率）に留める。
> 関連：[03 研究知見](03-research-findings.md) §5 データ資産 / `handoff_for_data.md`。

実装：`data/sources/edinet.py`・`edinet_taxonomy.py`／`equities/edinet_fundamentals.py`・
`edinet_factors.py`／`examples/download_edinet.py`・`edinet_validate_slice.py`・
`diag_edinet_fundamentals.py`。テスト：`tests/test_edinet_fundamentals.py`（13 件）。

---

## 1. 取得アーキテクチャ

**主＝公式 EDINET API v2（無料・商用可）／補助＝edinetdb.jp（Free/Light, 100req/日）。**

- 書類一覧 `GET /documents.json?date=…&type=2`（メタデータ・PIT アンカー＝`submitDateTime`）を
  営業日ごとに by-date ミラー（`data/edinet/list/{YYYYMMDD}.parquet`・**追記専用**＝初回観測を保全）。
- 書類取得 `GET /documents/{docID}?type=5`（XBRL→CSV・**ZIP 内 UTF-16 タブ区切り**・
  要素ID/コンテキストID/値）。`data/edinet/docs/{docID}_5.zip`。成功/失敗は Content-Type 判定。
- レート：数値上限は非公表。**1 リクエスト 3〜5 秒間隔**＋指数バックオフ（`EDINET_MIN_INTERVAL`）。
- **ローリング 10 年窓**：公式の保持は前方約 10 年。最古年（2016 提出＝FY2015）は窓の縁で
  日々取得不能になる → `download_edinet.py` は **最古年優先（periodEnd 昇順）**でバックフィルし、
  落ちる前に確保する。窓外の数日は 404（無害）。
- 大規模取得は **手元ターミナルで直接実行**（`examples/download_edinet.py`・進捗・中断再開可）。
- 補助 edinetdb.jp（`edinetdb.py`・`X-API-Key`・100/日）は **クロスチェックと公式窓外の古年
  （FY2012-2015）穴埋め**に限定（バルク不可）。**セグメント専用エンドポイントは Free REST に無い**（§9）。
- データ（`data/edinet/`）は **gitignore**。コミット可はコード・マッピング・クロスウォークのみ。

---

## 2. 要素ID → 正準フィールドのマッピング（基準別）

会計基準は要素IDの名前空間 prefix で表れる：**`jppfs_cor`＝JGAAP 本表／`jpigp_cor`＝IFRS
（要素は "IFRS" 接尾）／`jpcrp_cor`＝共通開示（主要な経営指標等・R&D）／`jpdei_cor`＝書類情報**。
US-GAAP 企業は本表が銘柄固有拡張中心で、標準タグを持たない。

**コンテキスト選択（最重要）**：連結＝`CurrentYearDuration`/`CurrentYearInstant`（メンバー無し）、
親会社単体＝`_NonConsolidatedMember` 付き。表示が文字化けする「連結・個別」列ではなく
**コンテキストIDで選択**する（`consolidated_current`）。要素IDは年度・企業でゆれるため候補を
**優先リスト**で持ち、売上等は **接尾辞フォールバック**で銘柄固有拡張も拾う。

| 正準フィールド | IFRS（`jpigp_cor:`） | JGAAP（`jppfs_cor:`） | 共通/備考 |
|---|---|---|---|
| net_sales | `RevenueIFRS`／`NetSalesIFRS`／… ＋接尾辞FB | `NetSales`／`OperatingRevenue1` | 銘柄固有拡張あり（例：トヨタ `TotalNetRevenuesIFRS`） |
| operating_income | `OperatingProfitLossIFRS` | `OperatingIncome` | IFRS は非標準＝日立等は独自拡張→null |
| ordinary_income | —（IFRS に無し） | `OrdinaryIncome` | |
| pretax_income | `ProfitLossBeforeTaxIFRS` | `IncomeBeforeIncomeTaxes` | ROIC 用 |
| profit | `ProfitLossAttributableToOwnersOfParentIFRS` | `ProfitLossAttributableToOwnersOfParent` | 親会社株主帰属 |
| total_assets | `AssetsIFRS` | `Assets` | |
| net_assets | `EquityIFRS` | `NetAssets` | **= J-Quants `Eq`（NCI 込）** |
| equity | `EquityAttributableToOwnersOfParentIFRS` | `ShareholdersEquity` | 親会社株主持分（NCI 除く） |
| gross_profit | `GrossProfitIFRS` | `GrossProfit` | IFRS は無い場合 null（商社/金融） |
| cfo / cfi / cff | `NetCashProvidedByUsedIn{Operating,Investing,Financing}ActivitiesIFRS` | 同名 ＋ `InvestmentActivities`（表記ゆれ） | 任天堂は Investment |
| cash | `CashAndCashEquivalentsIFRS` | `CashAndCashEquivalents` | |
| depreciation | `DepreciationAndAmortizationOpeCFIFRS` | `DepreciationAndAmortizationOpeCF` | EBITDA 用 |
| income_taxes | `IncomeTaxExpenseIFRS` | `IncomeTaxes`（current+deferred） | 実効税率 |
| interest_debt | 集約 `InterestBearingLiabilities{CL,NCL}IFRS` 優先→個別 `BondsAndBorrowings`/`Borrowings`/`LongTermDebt`/`BondsPayable`/`LeaseLiabilities` 合算 | `ShortTermLoansPayable`＋`LongTermLoansPayable`＋`BondsPayable`＋`CommercialPapers`＋`LeaseObligations`… の和 | 二重計上回避・JGAAP 無借金=0 |
| rd_expense | `jpcrp_cor:ResearchAndDevelopmentExpensesResearchAndDevelopmentActivities`（共通） | 同左 | 疎 |
| shares_outstanding | `jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults`（単体ctx 可） | 同左 | 発行済株式総数 |

**コードクロスウォーク**：EDINET `secCode`（5 桁）も J-Quants `Code`（5 桁）も同形式＝**恒等で結合**。

---

## 3. PIT 統合（提出日アンカー・FYE 非依存）

`edinet_fundamentals.build_edinet_long()` がキャッシュ済み有報を増分パースし、正準フィールドの
長形式（`Code`・`DiscDate`=提出日時・`period_end`・`basis`・各フィールド）を作る。これを
**既存の `fundamentals.point_in_time`（DiscDate ≤ t − lag の最新開示のみ採用）にそのまま流す**
＝ J-Quants fins_summary と同一の PIT 機構を再利用し、`edinet_fundamentals_panel()` で
as-of wide パネルにする。決算期末・報告義務日は使わず提出日のみをアンカーにするため
**FYE が銘柄ごとに不均一でも整合**する（訂正報告は訂正提出日以降にのみ反映）。

- **基準タグ**：`basis` 列（IFRS/JGAAP/USGAAP）。基準別 null は呼び出し側で扱う。
- **会計基準移行の断絶**：`add_basis_transition` が前期から基準が変わった開示に `basis_changed` を立て、
  移行年（および会計年度が非連続な年）をまたぐ YoY（資産成長・純株式発行）を **NaN 化**する
  （遡及再表示で BS が不連続になるため壊れた成長率を持ち込まない）。

---

## 4. 特徴量（§3）

`edinet_factors.py`。ファンダのみで決まる比率・前年比は **開示レベル**で会計年度整合に算出してから
as-of サンプル（YoY は移行/非連続で NaN）。価格依存（利回り）のみ時価総額と合成。各特徴量は
**raw** を出力し、`cross_sectional_zscore`／`cross_sectional_rank`（[-1,1]）／`sector_neutralize`
を重ねて 3 ビューを得る。**加工済み比率は使わず生ライン項目から自前計算**。

| 特徴量 | 定義 | 備考 |
|---|---|---|
| fcf_yield / fcf_yield_3y | FCF / 時価総額（**FCF = 営業CF＋投資CF に固定**） | 3 年平均版で M&A の振れを平滑・`ma_year` フラグ |
| cf_to_price | 営業CF / 時価総額 | |
| gross_profitability | 売上総利益 / 総資産 | Novy-Marx 2013 |
| ebitda_margin | (営業利益＋減価償却) / 売上 | |
| roic | 営業利益×(1−実効税率) / (有利子負債＋純資産) | 税率は [0,1] にクリップ |
| asset_growth | 総資産の前年比 | investment 軸（低成長=プレミアム）・移行/非連続は NaN |
| accruals | (純利益 − 営業CF) / 総資産 | 簡易 CF ベース（低い=高品質） |
| leverage | 有利子負債 / 純資産 | D/E |
| net_share_issuance | 発行済株式数の前年比 | **分割調整は未実施（既知の限界）** |
| rd_intensity | 研究開発費 / 売上 | 疎 |

---

## 5. 被覆率（バックフィル進行中・`diag_edinet_fundamentals.py` 出力）

最古年優先のバックフィルが進むほど埋まる。下表は途中経過の一例（FY2015 を 2016 提出分で捕捉）：

- **年×基準（開示件数）**：2016 提出＝FY2015 が JGAAP 285／US-GAAP 38（最古年確保が機能）。
  直近年（2023–2026）は IFRS 中心。完走で全上場×約 10 年に拡大。
- **フィールド充足率**：総資産・純資産・CFO・CFI・有利子負債・税金 ≈ 88%、売上 85.6%、
  営業利益 85.0%、粗利益 78.8%、**R&D 3.2%（疎）**。null は会計基準差（IFRS の営業利益/粗利益
  欠落・US-GAAP 未マッピング）と任意開示（R&D）に対応。

---

## 6. 突合（マッピングの妥当性検証）

`edinet_validate_slice.py`（会計基準を跨ぐ 10 社）で J-Quants fins_summary と突合：
**89 一致 / 1 警告 / 10 欠落**。`diag_edinet_fundamentals.py` の全キャッシュ集計でも各項目
OK 多数・WARN 1・MISS 2 で一貫。**IFRS/JGAAP の本体値は完全一致**。突合基準は |相対差|>1% で WARN。

- WARN：ソニー売上（売上 vs 売上＋金融事業収入の定義差）。
- MISS：キヤノン（US-GAAP・別タクソノミ未マッピング）、日立（IFRS の営業利益＝独自拡張）。

---

## 7. 既知の限界

1. **US-GAAP（キヤノン等少数）**：別タクソノミで標準タグを持たず未マッピング → 当該銘柄は
   J-Quants 値で代替するか US-GAAP マッピングを後日追加（`basis="USGAAP"` で識別可）。
2. **IFRS の営業利益・売上総利益は非標準**：IFRS は「営業利益」を強制しない。日立等は独自拡張、
   商社/金融は売上総利益なし → 当該フィールド null（ROIC/EBITDA/粗利益性も連動して NaN）。
3. **interest_debt の軽度過少**：銘柄固有の満期内訳拡張（例：日立 `CurrentPortionOfLongTermDebt`）は
   拾えず一部過少。IFRS で集約・個別タグとも無い場合は **None（偽ゼロを出さない）**。JGAAP の
   無借金（任天堂・キーエンス）は 0。
4. **net_share_issuance の分割調整未実施**：raw 発行株数の前年比のため分割年は過大に出る
   （J-Quants `AdjustmentFactor` での補正は今後）。
5. **セグメント系は未実装**：有報注記のパースが必要（保留）。補助 edinetdb の Free REST にも
   セグメント専用エンドポイントは無い。
6. **ローリング 10 年窓**：最古年は逐次取得不能化。バックフィルは最古年優先で対抗。
7. **FCF の M&A 歪み**：FCF は買収年に大振れ → 3 年平均＋`ma_year` フラグで対応。
8. **訂正報告**：原報告を原提出日で採用し、訂正は訂正提出日以降にのみ反映（point_in_time が
   同一開示日の最後を採用）。

---

## 8. アンチ p-hacking（K 不変）

本タスクはデータ基盤整備であり戦略探索ではない。`judge_grid`・永続レジストリは一切触らず、
予測力確認は `diag_edinet_fundamentals.py` の **throwaway 診断（月次IC・被覆率・突合）**に留める。
新規 DSR 判定は Phase 4 で一度だけ。`examples/registry_status.py` で K 不変を確認できる。

---

## 9. 補助ソース edinetdb.jp（`data/sources/edinetdb.py`）

公式 EDINET（一次）の **独立クロスチェック** と **古年穴埋め** に限定した補助。**バルク不可**
（Free=100req/日）。ベース `https://edinetdb.jp/v1`・認証ヘッダ `X-API-Key`（`.env` の
`EDINET_DB_API_KEY`）・チャット側 MCP には依存しない。**日次クォータを永続カウント**
（`data/edinet/edinetdb/quota.json`・既定上限 95）し、上限で `QuotaExceeded`（翌日に回す）。
キャッシュ命中は quota を消費しない。

- `/companies/{EDINETコード}/financials`：**FY2012〜最新**の時系列（accounting_standard・revenue・
  operating_income・net_income・total_assets・net_assets・cf_*・shares_issued・split_adjustment_factor 等）。
  **生ライン項目のみ**採り、加工済み比率（roe_official 等）は使わない。
- **クロスチェック実測**：トヨタ FY2023-2026 × 10 項目で **比較可能 36 項目すべて rel_diff=0.0 で
  完全一致**（公式パースの独立検証）。不一致は ordinary_income のみ＝IFRS に経常利益が無いための
  定義差（公式 NaN／edinetdb は別流儀で保持）。`diag_edinet_fundamentals.py --edinetdb N` で
  N 銘柄を 3-way（official／edinetdb／J-Quants）突合できる。
- **古年穴埋め**：公式の ~10 年窓（2016+）より古い **FY2012-2015** を `old_year_backfill` で補える。
- **コードクロスウォーク**：実コードは EDINET コード（E始まり）。secCode(5桁)→EDINET コードは
  一覧ミラーの edinetCode↔secCode（`seccode_to_edinet`・quota 不要）で変換。
- 限界：segments・interest_bearing_debt は Free REST に無い。net_assets は edinetdb=自己資本
  （owners）で公式の total equity と定義が異なるため突合対象外。
