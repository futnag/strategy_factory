# 作業依頼：特徴量拡充 — 信用/空売り・微細構造（A）＋ fins_summary 新ファンダ（B）

## 0. 位置づけ

`claude_local_sessions`（strategy_factory）を GKX（Gu, Kelly & Xiu 2020）型 ML×クロスセクション予測へ育てる計画の、**特徴量拡充の続き**。Phase 1（価格/流動性）・Phase 2（EDINET 三表）に続き、**EDINET バックフィルの完走を待つ間に並行で**、既存の J-Quants データを使って GKX 特徴量セットをさらに広げる。本タスクは 2 ワークストリーム：

- **A：信用/空売り・微細構造** — すでにダウンロード済みで、ヘルパー関数も在る（`equities/margin.py`・`features/microstructure.py`）のに、**月次 PIT のクロスセクション特徴量としては未材化**のデータを起こす。価格系と直交する情報を足せるのが狙い。
- **B：fins_summary 新ファンダ** — 決算サマリー（J-Quants・EDINET 非依存）から、既存の value/quality/size に無い特徴量（SUE/PEAD・予想改訂・成長・安定度・持続可能成長・52週高値乖離・季節性・Dimson β）を追加。

いずれも将来の ML 推定器（Phase 4）の入力＝**1 仮説のインフラ**であり戦略探索ではない。**K（永続レジストリ）は消費せず・判定せず・診断は throwaway**。

---

## 1. スコープ

### やること
- A・B の特徴量を、既存ヘルパー・標準化ユーティリティ・PIT 機構を**再利用**して実装し、`feature_store` で**月次 PIT・float32・wide**に材化、raw／[-1,1]ランク／セクター中立の 3 ビューを供給。
- テスト（先読み不変ほか）・docs・throwaway 診断を追加。

### やらないこと
- **投資部門別フロー（`flows.py`）は対象外**。これは銘柄別でなく市場区分レベルの系列なので、Phase 4 の「特性×マクロ」**状態変数**として別途扱う（本タスクで per-stock 特徴量にしない）。
- **EDINET 由来データには触れない**（バックフィル実行中・Phase 2 完了済。本タスクは EDINET 非依存で完結）。
- ML 推定器・特性×マクロ交互作用・DSR 判定・新規戦略登録は行わない。

---

## 2. 既存資産（在るもの＝再利用／再実装禁止）

**A-1 信用/空売り（`equities/margin.py`）**：ローダ（`load_weekly_margin`/`load_short_ratio`/`load_short_positions`）と派生関数が既に在る。
- `margin_imbalance(weekly)` → `[Date, Code, margin_imbalance, short_to_long]`
- `short_interest(positions)` → `[Date, Code, short_interest]`（CalcDate 基準・対 SO 比率合算）
- `sector_short_ratio(ratio)` → `[Date, S33, sector_short_ratio]`

**A-2 微細構造（`features/microstructure.py`）**：OHLCV のみで計算する Series 関数が既に在る。
- `parkinson_vol(high, low, w)` / `garman_klass_vol(o,h,l,c,w)`（レンジ・ボラ）
- `roll_spread(close, w)` / `corwin_schultz_spread(high, low)`（実効スプレッド推定）
- `vpin(close, volume, …)` / `rsi(close, w)`
- ※`amihud_illiquidity` は**既に price_factors で月次材化済**（`amihud_illiq`）。**重複追加しない**。

**B 共通**：`equities/factors.py` の `value_quality_size_factors`（earnings_yield/B-M/cf_yield/sales_yield/div_yield/**roe/roa/op_margin/equity_ratio/accruals**/size/momentum）と標準化ユーティリティ（`cross_sectional_zscore`/`cross_sectional_rank`[-1,1]/`sector_neutralize`）、`equities/fundamentals.py: fundamentals_panel()`（提出日アンカーの PIT as-of）。**既存ファンダは再実装しない**。

**材化パターン**：`data/feature_store.py` の `build_price_liquidity_features`（日次計算→月末 as-of→float32 wide）と `price_factor_view`（3 ビュー）を踏襲。`equities/universe.point_in_time_universe`（生存者バイアス除去）上で評価。Silver は `data/store.py: load_wide`。

---

## 3. 最重要の規律

1. **PIT・先読み厳禁**：各 `f[t]` は ≤t の情報のみ。trailing 窓は過去方向。**各特徴量に「未来改変→≤t 不変」テスト必須**（price_factors の `test_lookahead_invariance` と同じ流儀）。
2. **信用/空売りは「公表日アンカー」**（最重要・EDINET の提出日アンカーと同一思想）：週末信用残高・大量空売り残高は**残高基準日ではなく、公開され利用可能になった日**で ≤t 採用する。週次信用は基準日の数営業日後に公表、大量空売り（short_positions）は T+2 開示。メタに公表日があればそれを、無ければ**既知の公表ラグで保守的にラグ**。基準日を直接使わない。
3. **符号の扱い**：
   - **方向が確立した因子は house 符号「大きいほどロング側」を適用**：レンジ・ボラ（低ボラ＝負号）、スプレッド推定（非流動性プレミアム＝正号）、52週高値乖離（正）、SUE/PEAD（正）、利益安定度（高安定＝ロング）。
   - **方向が係争的な信用需給は raw 向きを維持**（`margin.py` が「符号仮説は中立」と明記）＝ハード反転せず、docs に literature 符号仮説を併記。`short_interest` のみ空売りアノマリー（高 SI→低リターン）に沿って**負号を推奨**（docs に根拠）。実証符号は throwaway IC 診断で確認。
4. **K 不変・反 p-hacking**：データ基盤であり戦略探索ではない。**`judge_grid` 登録・判定しない／レジストリ不変**。予測力確認は throwaway 診断（IC・被覆・冗長性相関）に留める。判定は Phase 4 で一度だけ。
5. **窓・パラメータは §4/§5 標準で事前固定**（チューニング探索しない）。
6. **EDINET 非依存**：バックフィル未完でも本タスクは green であること。

---

## 4. ワークストリーム A：追加特徴量

### A-1 信用/空売り（月次 PIT 化）
`margin.py` の派生を long→wide（index=月末, col=Code）に整形し、**公表日アンカー**で月末 as-of サンプリング、float32 材化：
- `margin_imbalance`、`short_to_long`（raw 維持・符号中立）
- `short_interest`（負号推奨・docs に根拠）
- `sector_short_ratio`：S33 レベル → 構成銘柄へブロードキャスト（`price_factors.industry_momentum` と同じ方式・sector-PIT 限界は docs に明記）
- 追加で容易なもの：`margin_balance_change`（信用残の前週比）、days-to-cover 近似（売残/平均出来高）。標準窓は事前固定。

### A-2 微細構造（クロスセクション適用＆材化）
`microstructure.py` の Series 関数を**全銘柄に適用**（列方向 apply かベクトル化）し、月末 as-of・float32 材化。**入力は分割調整済 OHLC**（分割日のジャンプでスプレッド/レンジ推定が壊れるため。Silver の調整済 high/low/close、無ければ調整係数を適用）：
- `parkinson_vol` / `garman_klass_vol`（窓=20。**低ボラ＝負号**）
- `roll_spread`（窓=20）/ `corwin_schultz_spread`（**非流動性プレミアム＝正号**）
- `vpin`（窓=50・raw・[0,1]）、`rsi`（窓=14・raw・[0,100]。過熱/反転の素材）
- ※`amihud` は既存ゆえ追加しない。

---

## 5. ワークストリーム B：fins_summary 新ファンダ（EDINET 非依存）

`fundamentals_panel()`（提出日アンカー PIT）＋日次 adj_close から。**既存の roe/roa/op_margin/accruals/earnings_yield 等は再実装しない**。新規のみ：
- **SUE / PEAD**：標準化サプライズ（実績 EPS − 直近会社予想 FEPS）/ スケール。発表後の持続（PEAD）＝正号。
- **予想改訂**：会社予想 FEPS の改訂率を特徴量化（既存 PEAD シグナルの一般化）。
- **成長**：売上成長・利益成長・**自己資本成長**（※資産成長の代理。EDINET 総資産ベースの investment 軸とは別物と docs に明記）。
- **安定度（クオリティ）**：過去 ROE・マージンの trailing ボラティリティに負号（高安定＝ロング）。
- **持続可能成長率**：ROE×(1−配当性向)。
- **52週高値乖離**（George-Hwang 2004）：close / trailing 252 営業日 max（正号）。日次のみ。
- **季節性**（Heston-Sadka 2008）：過去複数年の同暦月リターン平均。日次のみ。
- **Dimson β**（薄商い対応）：当期＋ラグ市場リターンへの回帰係数和。`price_factors._rolling_beta_residvar` を拡張 or 併用（既存 beta と整合）。

---

## 6. 成果物（Deliverables）

1. **A-1**：`equities/margin.py` を拡張（or `equities/holdings_factors.py` 新設）して、派生を月次 PIT wide 化する関数群（公表日アンカー込み）。
2. **A-2**：`data/feature_store.py` に `build_microstructure_features`（microstructure.py をクロスセクション適用→月末材化）。
3. **B**：`equities/factors.py` 拡張（or `equities/fundamental_factors.py` 新設）で §5 を `fundamentals_panel`＋adj_close から実装。
4. `data/feature_store.py` に `build_holdings_features` / `build_fundamental_features_v2` を追加し、`materialize_features` に編入（再計算で冪等）。3 ビューは `price_factor_view` を拡張 or 同等関数で供給。
5. **テスト**（`tests/`）：①先読み不変（各新特徴量・未来改変で ≤t 不変）、②**公表日アンカーの PIT**（基準日でなく公表日で乗ること・残高更新が未来に漏れないこと）、③被覆率、④微細構造の調整済 OHLC 使用（分割日でスプレッドが発散しない）、⑤3 ビュー境界。現行 391 件に追加し全 green。
6. **docs**（例 `docs/16-holdings-microstructure-factors.md`・`docs/17-fundamental-factors-v2.md`）：各特徴量の算式・データ源・窓・符号（係争中は中立と明記）・公表日ラグ・既知の限界（信用/空売りの sector-PIT・SUE のスケール定義・季節性のデータ要件）。docs/03 §5 からリンク。
7. **診断（throwaway・K 不変）**：新特徴量の月次 IC・被覆・**既存 factor との冗長性相関**（特に A-2 と既存 amihud/rvol、B の成長と既存 momentum）。`examples/diag_factor_expansion.py`。判定でない旨を明記。
8. データ追加なし（既存キャッシュのみ）。`examples/registry_status.py` で **K がタスク前後で不変**を確認。

---

## 7. 受け入れ基準（Definition of Done）

- A-1・A-2・B の各特徴量が PIT・float32・wide で材化され、PIT ユニバース上で月末断面に抽出できる。
- 信用/空売りが**公表日アンカー**で乗り、基準日直接参照や未来漏れがテストで否定されている。
- 微細構造は**調整済 OHLC** で計算され、分割日の発散が無いことがテストで確認できる。
- 先読み不変・被覆・3 ビュー境界・公表日 PIT のテストが全 green（既存 391 件も不変）。
- 既存因子（amihud/rvol/roe/roa/accruals/momentum）を**再実装していない**。
- docs に算式・符号・公表日ラグ・限界が揃う。
- 診断 IC・冗長性レポートが出力でき、**K 不変**。**EDINET 非依存**で完結（バックフィル未完でも green）。

---

## 8. 運用上の留意点

- 純 Python・OS 非依存・`pathlib`・LF・float32・pyarrow を維持。
- 既存（margin.py・microstructure.py・fundamentals_panel・標準化ユーティリティ・PIT ユニバース・feature_store の材化パターン）を**再利用**し重複実装を避ける。
- 符号は §3-3 の規約（確立済みは house 符号・係争中の信用需給は raw 維持・short_interest のみ負号推奨）を厳守し、docs に根拠を残す。
- 不明点（週次信用/大量空売りの公表日メタの有無と公表ラグ・Silver に調整済 high/low があるか・SUE のスケール定義・季節性に必要な過去年数）は、推測で固めず実データのスキーマを確認し、判断が要る箇所は質問する。
- flows.py（投資部門別）は本タスク対象外（Phase 4 の状態変数）。

---

### 一言サマリ（Claude Code へ）
> EDINET バックフィルの完走を待つ間に、**既にダウンロード済みなのに未 factor 化のデータ**を GKX 特徴量に起こしてほしい。**A**：`margin.py`（信用需給・空売り）と `microstructure.py`（Parkinson/Garman-Klass ボラ・Roll/Corwin-Schultz スプレッド・VPIN・RSI）を、月次 PIT・[-1,1]ランク・セクター中立で材化（信用/空売りは**公表日アンカー**、微細構造は**調整済 OHLC**、amihud は既存ゆえ重複禁止）。**B**：fins_summary から SUE/PEAD・予想改訂・成長・安定度・持続可能成長・52週高値乖離・季節性・Dimson β を追加（既存 roe/roa/accruals 等は再実装しない）。符号は確立済みのみ house 規約で付け、係争的な信用需給は raw 維持＋docs 記載。これはデータ基盤なので **K 不変・判定なし・診断は throwaway**、PIT 先読み不変テストは必須、**EDINET 非依存**で green に。flows は対象外（Phase 4 状態変数）。