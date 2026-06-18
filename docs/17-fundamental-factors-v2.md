# 17. fins_summary 新ファンダ ファクター（GKX 特徴量拡充 B）

決算サマリー（J-Quants・**EDINET 非依存**・提出日アンカー PIT）から、既存 value/quality/size に無い
特徴量を追加する（SUE/PEAD・予想改訂・成長・安定度・持続可能成長・52週高値・季節性・Dimson β）。
**既存の roe/roa/op_margin/accruals/earnings_yield 等は再実装しない**。Phase 4 の入力＝K不変・判定
なし・診断 throwaway。

実装：`equities/fundamental_factors.py`（純関数）／`data/feature_store.py:
build_fundamental_features_v2`（月次 PIT 材化）。`fundamentals.point_in_time`（DiscDate アンカー）・
adj_close を再利用。テスト：`tests/test_factor_expansion.py`。関連：[03 §5](03-research-findings.md)。

---

## 1. 特徴量カタログ

開示レベル（fins_summary 由来）の特徴は会計年度整合に算出 →
`point_in_time(date_col="DiscDate", lag_days=1)` で月末 as-of。価格依存（52週高値・Dimson β）と
季節性は日次/月次。

| 特徴量 | 定義 | 符号 |
|---|---|---|
| sue_recent | (実績EPS − **当該FYの直近(3Q)会社予想FEPS**) / 月末終値 | 正（サプライズ利回り） |
| sue_initial | (実績EPS − **当該FYの期初(1Q)会社予想FEPS**) / 月末終値 | 正 |
| forecast_revision | 全年度予想 FEPS の前回開示差 / 月末終値 | 正（上方修正→ロング） |
| sales_growth / profit_growth | 売上・純利益の前年比（FY） | raw（方向中立） |
| equity_growth | 自己資本の前年比（FY） | raw（※資産成長の代理。EDINET investment 軸とは別物） |
| roe_stability / margin_stability | 過去 ROE・営業利益率の trailing 標準偏差に**負号**（最低3期） | 正（高安定＝ロング） |
| sustainable_growth | ROE×(1−配当性向)。PayoutRatioAnn は**比率**（実データ確認・÷100 不要） | raw |
| high_52w | adj_close / trailing 252 営業日 max | 正（George-Hwang 2004） |
| seasonality | 過去複数年の**同暦月**月次リターン平均（最低5年） | 正（Heston-Sadka 2008） |
| dimson_beta | 当期＋ラグ市場への（単回帰）β係数和に**負号** | 正（低ベータ＝ロング・Dimson 1979） |

**SUE 連携（CurFYEn）**：FY 開示の実績 EPS と、当該 FY（CurFYEn）の四半期開示が持つ全年度予想 FEPS を
突合。直近予想＝最後の四半期(3Q)、期初予想＝最初(1Q)。日本企業は期中に予想を実績へ寄せるため
**直近差は小さくなりがち**（例：トヨタ FY2025 は直近差 +18.7 円 vs 期初差 +94.5 円）→ recent と
initial の**両方**を供給（ML 入力は増やしてよい）。年次サプライズなので PEAD の持続は月次 as-of が
次開示まで値を持ち越すことで拾う。

**スケール（株価除し）**：surprise yield は赤字でも頑健・分母が常に正でゼロ近傍にならない（PEAD 文献の
標準 deflator）。winsorize は zscore/rank ビューが担う（rank はランク化で外れ値不感）。

---

## 2. 3 ビューと標準化

raw を材化し、`feature_store.price_factor_view(name, view)` で raw/zscore/rank([-1,1])/sector_neutral を
既存ユーティリティから生成（factors.py 再利用）。SUE/安定度の raw は penny・微小資本銘柄で外れ値が
出るが、zscore（±3σ winsor）/rank（順位）と PIT ユニバース上位500 が吸収する。

---

## 3. 診断（`examples/diag_factor_expansion.py`・throwaway・K不変）

PIT ユニバース上位500・121月・翌月リターンの月次 Spearman IC：

- **効くもの**：roe_stability +0.033（IR **0.25**）、high_52w +0.033（IR 0.18）、forecast_revision
  +0.021（IR 0.24）、margin_stability +0.018。
- **弱い**：sue_recent/sue_initial ≈ 0（**年次 SUE は月次粒度では弱い**＝PEAD は本来日次/イベント
  効果。docs/03 の PEAD 知見と整合）、profit/sales_growth・sustainable_growth ≈ 0、equity_growth
  −0.018（高成長→低リターン＝investment 方向）、dimson_beta −0.021。
- **冗長性**：dimson_beta↔beta **+0.94**（上位500の流動銘柄では薄商い補正がほぼ効かず ≒ 当期β。
  Dimson の価値は小型・薄商いにある）、high_52w↔mom_6m +0.57。

---

## 4. 既知の限界

- **SUE は年次粒度**：四半期 SUE（約20点）でなく年次（~10点）。月次サンプリングでは PEAD が弱い。
  日次/イベント実装は別途（docs/03 の PEAD 系知見参照）。
- **季節性は履歴 ~10 年で短い**（本来 20 年級）：同暦月の観測が最大 ~10 点でノイジー＝**弱い実験的
  特徴**。最低5年の床を課す。Phase 5 の履歴延伸で改善余地。
- **equity_growth は資産成長の代理**だが EDINET 総資産ベースの investment 軸とは別物（自己資本は
  利益留保・増資・自己株で動く）。混同しないこと。
- **dimson_beta は係数和の近似**（厳密な重回帰でなく当期β＋ラグβの和・ベクトル化のため）。流動
  ユニバースでは当期βとほぼ重複（§3）。
- **SUE/安定度の raw 外れ値**：penny・微小資本で発散しうる→ zscore/rank ビューと PIT ユニバースで
  吸収（§2）。
- **PayoutRatioAnn は比率前提**（実データで確認）。仕様変更時は要再確認。

---

## 5. アンチ p-hacking（K不変）

データ基盤整備であり戦略探索ではない。`judge_grid`・永続レジストリは触らず、予測力確認は
`diag_factor_expansion.py` の throwaway 診断に留める。新規 DSR 判定は Phase 4 で一度だけ。
`examples/registry_status.py` で K 不変を確認。
