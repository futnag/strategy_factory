# 作業依頼：価格系・流動性系の特徴量ライブラリ拡充（Phase 1・現状反映版）

## 0. このプロンプトの位置づけ

`claude_local_sessions`（strategy_factory）を GKX（Gu, Kelly & Xiu 2020）型の ML×クロスセクション予測へ育てる計画の **Phase 1**。Phase 2（EDINET 三表の PIT 統合・`edinet_factors.py` 等）は完了済で、いま EDINET バックフィルを手元で実行中。**本タスクは、その完走を待つ間に並行で進められる「価格系・流動性系の特徴量拡充」**。

重要な前提：現状の `equities/factors.py` は **ほぼ当初の十数個のまま**で、GKX が最も支配的と示した**価格系（モメンタムの多ホライズン・短期リバーサル・各種ボラ・特異ボラ・ベータ・MAX）と流動性系（Amihud・回転率・売買代金・ゼロリターン日）がごっそり欠けている**。Phase 4（ML推定器）に進む前に、この欠けを **追加データコストゼロ**（既存の J-Quants 日次データのみ）で埋めるのが狙い。

これらは将来の ML 推定器（Phase 4）の入力であり**戦略探索ではない**。Phase 2 と同じく **K（永続レジストリ試行数）は消費せず・判定せず・診断は throwaway**。

---

## 1. スコープ

### やること
- 既存の **J-Quants Standard 日次データ（調整済価格 AdjC・生株価 C・売買代金 Va・発行済/自己株式数）＋市場リターン系列＋S33業種**だけで作れる **価格系・流動性系の特徴量**を §4 に従って追加する。
- 既存の標準化ユーティリティ（`cross_sectional_zscore` / `cross_sectional_rank`[-1,1] / `sector_neutralize`）を**再利用**して、raw／[-1,1]ランク／セクター中立の3ビューを供給する。
- `data/feature_store.py` の材化に組み込み、PIT・float32 の月次クロスセクション特徴量パネルとして出力する。
- テスト（先読み不変ほか）・docs・診断（throwaway）を追加する。

### やらないこと
- **EDINET 由来データには触れない**（Phase 2 完了済・バックフィル実行中なので非依存に保つ）。本タスクは EDINET バックフィルの完走を待たずに完結できる。
- ML 推定器（Phase 4）・特性×マクロ交互作用テンソルの生成・DSR 判定・新規戦略の登録は**行わない**。
- アナリスト系（Phase 3）・Premium 履歴延伸（Phase 5）は対象外。

---

## 2. 既存資産（在るもの＝重複実装しない／再利用する）

`equities/factors.py` に既に在るもの（**再実装禁止**）：
- バリュー/クオリティ/サイズ：`value_quality_size_factors()` が earnings_yield(予想E/P)・book_to_market・cf_yield(営業CF利回り)・sales_yield・div_yield・roe・roa・op_margin・equity_ratio・accruals・size・**momentum(12-1のみ)** を生成。
- `low_volatility(price, window)`（実現ボラに負号・1種）
- `residual_momentum(ret, market, window=36, mom_len=12, skip=1)`（iMOM。市場単回帰の残差モメンタム）
- 標準化：`cross_sectional_zscore(df, winsor=3.0)` / `cross_sectional_rank(df)`→[-1,1] / `sector_neutralize(df, sector)` / `cross_sectional_residualize(target, controls)`
- `market_cap(raw_price, shares_out, treasury)`

`data/feature_store.py`（Gold）が材化済：returns・log_returns・**vol_20**・**momentum_12_1**・**reversal_5(5日)**・regime（float32）。

その他：`data/store.py`（Silver：フィールド別 wide。生/調整価格・Va・株数を back-adjust 再構築）、`equities/universe.py`（生存者バイアス除去の PIT ユニバース）、`equities/fundamentals.py`（PIT as-of）、`residual_momentum` が使う**市場リターン系列**（TOPIX/等加重/FF-JP のいずれか既存のもの）。

---

## 3. 最重要の規律（Phase 2 と同一）

1. **PIT・先読み厳禁**：各特徴量 `f[t]` は `≤t`（価格・出来高は ≤t、月次パネルなら ≤ 当月末）の情報のみ。trailing 窓は過去のみ参照。**各特徴量に「未来を改変しても ≤t 値が不変」を assert するテスト必須**。既存 `low_volatility`/`residual_momentum` の詳細な PIT docstring と同じ厳密さで書く。
2. **K不変・反p-hacking**：これはデータ基盤。**`judge_grid` で登録・判定しない／永続レジストリ不変**。予測力は IC・被覆・相関の**診断（throwaway・レジストリ不使用）**でのみ可視化。判定は Phase 4 で一度だけ。
3. **窓長・パラメータは §4 の標準で事前固定**。「良くなるまで窓を動かす」探索はしない（p-hacking 回避。docs/03 §6.6 の変種打ち切り規律と同じ）。
4. **生存者バイアス除去**：診断・評価は PIT ユニバース上で。固定ユニバース不可。
5. **ハウススタイル踏襲**：各 factor 関数は wide パネルを返しユニバース列へ reindex、不完全窓は NaN、**符号は「大きいほどロング側＝期待プレミアム方向」**に統一、docstring に「アノマリーの出典＋PIT注記＋符号根拠」を併記（既存 `residual_momentum` が Blitz-Huij-Martens 2011 / Chaves 2016・docs/04 を引くのと同じ流儀）。

---

## 4. 追加する特徴量（在るものは明記。標準窓は事前固定）

入力は日次（調整済価格・Va・mcap・市場系列・S33）。日次で計算し**月末断面に as-of サンプリング**して月次特徴量パネルへ（既存 `residual_momentum` の月次規約に合わせる）。符号は §3-5 のとおり。

### モメンタム／リバーサル
- **mom_1m_reversal**：過去1ヶ月リターンに負号（短期リバーサル）。※feature_store の reversal_5 は5日。月次1ヶ月版を追加。
- **mom_3m / mom_6m**：直近1ヶ月スキップの3/6ヶ月モメンタム。
- **mom_36m_reversal**：過去36ヶ月リターンに負号（長期リバーサル）。
- **industry_momentum**：S33業種の等加重トレーリングリターン（業種モメンタム）。
- ※ mom(12-1) は既存 `value_quality_size_factors` の `momentum` を流用（重複実装しない）。

### ボラティリティ／リスク
- **rvol_60 / rvol_252**：日次リターンの trailing 60/252 営業日標準偏差に負号（低ボラ・アノマリー）。※20日相当は feature_store の vol_20 が既存。
- **ivol**：市場（または FF-JP）への trailing 回帰残差の標準偏差に負号（特異ボラ）。市場系列は `residual_momentum` が使うものを再利用。残差化の自己条件付けに注意（leave-one-out かラグ推定）。
- **beta**：市場への trailing ベータ（低ベータ・ロング＝負号、BAB）。
- **max_ret**：過去1ヶ月の日次最大リターンに負号（MAXアノマリー）。
- **ret_skew**：過去 trailing 窓の日次リターン歪度。

### 流動性（現状まるごと欠落＝最優先で追加）
- **amihud_illiq**：trailing 窓の mean(|日次リターン| / 日次Va)（非流動性プレミアム＝正符号）。※小型・低流動性に偏在する点を docstring と docs に明記。
- **turnover**：trailing 平均 Va / 時価総額（高回転＝低リターン＝負符号）。
- **dollar_volume**：trailing 平均 Va の対数（流動性/サイズ代理）。
- **zero_ret_days**：trailing 窓のゼロリターン日比率（非流動性代理＝正符号）。

標準窓（事前固定）：短期=1ヶ月（≈21営業日）、ボラ=60/252営業日、ベータ/特異ボラ=trailing 252営業日（または既存 `residual_momentum` の window=36ヶ月と整合）、流動性=trailing 60営業日。確定値を docs に記録。

---

## 5. 成果物（Deliverables）

1. `equities/price_factors.py`（新規。`factors.py` 肥大化回避のため分割。standardize/neutralize は factors.py のものを import 再利用）に §4 の純関数を実装。各々 PIT・完全窓要求・NaN規約・符号・出典 docstring 付き。
2. `data/feature_store.py` 拡張：§4 を月次クロスセクションのwideパネル（float32）で材化。Silver（生/調整価格・Va・株数）と市場系列を入力に。**[-1,1]ランク版・セクター中立版**は既存ユーティリティで生成。
3. **テスト**（`tests/`）：①先読み不変（未来改変→≤t不変を assert）、②被覆率の健全性、③**既存材化との整合**（新 rvol が vol_20 と、新 mom_* が momentum_12_1 と、新 1ヶ月リバーサルが reversal_5 と整合的か）、④ランク/中立化の境界。現行382件に追加し全 green。
4. **docs**（例 `docs/15-price-liquidity-factors.md`）：各特徴量の算式・データ源・標準窓・符号・アノマリー出典・PIT注記・既知の限界（流動性系は小型偏在・取引コスト依存）。docs/03 §5 からリンク。
5. **診断レポート（throwaway・K不変）**：各特徴量の月次クロスセクション IC・被覆率・**新旧特徴量のペアワイズ相関**（冗長性の可視化：新モメンタム/ボラと既存の重複度。GKX は相関特徴量を1モデルに入れてよいが、明白な重複は把握）。`examples/diag_price_factors.py`。判定ではない旨を明記。
6. データ追加なし（既存 Silver/Gold のみ）。`examples/registry_status.py` で **K がタスク前後で不変**を確認。

---

## 6. 受け入れ基準（Definition of Done）

- §4 の価格系・流動性系が PIT・float32・wide で材化され、PIT ユニバース上で月末断面に抽出できる。
- [-1,1]ランク版・セクター中立版が既存ユーティリティで生成でき、欠損・外れ値の扱いが明文化。
- 先読み不変テストを含む新規テストが全 green（既存382件も不変）。
- 既存材化（vol_20 / momentum_12_1 / reversal_5）と新特徴量の整合がテストで確認できる。
- docs に特徴量カタログ（算式・窓・符号・出典・限界）が揃う。
- 診断 IC・相関レポートが出力でき、**永続レジストリ（K）が不変**。
- **EDINET 由来データに依存せず**完結している（バックフィル実行中でも本タスクは green）。

---

## 7. 運用上の留意点

- 純 Python・OS 非依存・相対パス/`pathlib`・LF・float32・pyarrow を維持。
- 既存（標準化ユーティリティ・PITユニバース・feature_store・市場系列）を**再利用**し重複実装を避ける。新規は §2 に無いものだけ。
- 符号規約「大きいほどロング側」を全 factor で統一し、docstring に根拠を書く（既存ハウススタイル）。
- 流動性系は小型・低流動性に偏在するため、docs に取引コスト/流動性フィルタとの相互作用の注意を明記（実装は Phase 4 の現実性評価で扱う）。
- 不明点（市場系列の具体的構築・月末as-ofの営業日整合・特異ボラの残差化方式）は、推測で固めず既存コードの規約を確認し、判断が要る箇所は質問する。

---

### 一言サマリ（Claude Code へ）
> GKX 化の Phase 1 を、**EDINET バックフィルの完走を待つ間に並行で**進めてほしい。現状 `factors.py` はほぼ当初の十数個で、**GKX で最も効く価格系（多ホライズン・モメンタム/短期リバーサル/各種ボラ/特異ボラ/ベータ/MAX）と流動性系（Amihud/回転率/売買代金/ゼロ日）が欠落**している。これを **既存の J-Quants 日次データだけ**（EDINET非依存）で `equities/price_factors.py` に追加し、`feature_store` で PIT・[-1,1]ランク・セクター中立の月次パネルとして材化する。既存の標準化ユーティリティと市場系列は再利用、既存 factor は再実装しない。これはデータ基盤なので **K不変・判定なし・診断は throwaway**、PIT 先読み不変テストは必須。窓長は §4 の標準で事前固定（チューニング探索はしない）。