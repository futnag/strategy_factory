# 接続設計 v2.1：日足価格・出来高レジーム検知を既存 `invest_system` へ載せる実装計画（コードは書かない）

> 本書は設計書 `regime_detection_design_v2.1.md` を既存リポジトリ `invest_system` に接続する**実装計画**。成果物は **①再利用/改修/新規 対応表 ②シーム定義（データ契約）③配置案 ④実装順序 ⑤広域指数フォールバック選択肢** まで。**実装コードは含めない**（擬似シグネチャ・データ契約まで可。釘③）。

## 0. 原則
1. **reuse-first**：既存基盤が成熟しているため、新規は「グルー＋日足層コア」に限定する。
2. **2つのPIT系を AsOfView で橋渡し**：既存には (A) `sector_regime` の walk-forward ループ（`_slice_window`/`refit_every`）と (B) `research/engine.py`＋`data_view.py:AsOfView` の2系がある。**レジーム検知は A 系で生成 → filtered 系列を PIT 時系列として B 系の `AsOf` フレームに登録 → 戦略/バックテストは B 系**、と段で分ける。シーム（§2）の不変条件はすべて `AsOf` 可用日で表現する。
3. **確定済みアーキ**：方向性＝業種指数（S33代理始動）／日足＝流動性ティア partial pooling（Amihud主・ADV副ゲート・弱め固定）／週足整列＝モメンタム符号。

---

## 1. 再利用 / 改修 / 新規 対応表

### 1.1 再利用（直接・改修なし）

| 設計要素 | 既存資産（file:func） | 用途 |
|---|---|---|
| 日足 OHLCV/出来高/代金/値幅フラグ | `data/store.py:load_wide("close"/"volume"/"turnover"/"upper_limit"/"lower_limit")` | §4.1 入力・**§4.3 値幅制限フラグ** |
| Amihud・レンジボラ | `features/microstructure.py:amihud_illiquidity, parkinson_vol, garman_klass_vol` | §4.1 |
| PIT 価格・流動性ファクタ（stale代理含む） | `equities/price_factors.py:amihud_illiq, dollar_volume, realized_vol, zero_ret_days, max_ret, skew` | §4.1・**§4.3 stale=`zero_ret_days`** |
| ADVゲート/流動性ユニバース（PIT） | `equities/universe.py:point_in_time_universe, liquid_universe_mask, LIQUID_UNIVERSE_PRESETS` | **§5.4 ティア二段①ゲート** |
| バックテスト・コスト・ターンオーバー | `research/engine.py:backtest(costs_bps=DF, execution_lag, adv, participation, no_buy/no_sell, short_borrow_bps, rebalance_band)` → `BacktestResult(returns, gross, turnover, ...)` | §6.4/§6.5・**§6.4 セル別コスト＝レジーム依存コスト** |
| 分位 L/S（横断派生時） | `equities/backtest.py:long_short_returns` | §6（任意） |
| 正準 PIT 機構 | `research/data_view.py:AsOfView, AsOf.frame()` | **シーム強制（§2）** |
| 指標・多重検定・CPCV | `research/phase4_evaluate.py:oos_r2, monthly_ic, ls_portfolio_returns, sharpe_ratio`／`validation/dsr.py:PSR,DSR`／`backtest/cpcv_backtest.py` | §6.4/§6.5・§2.8 |
| PEAD シグナル（PIT, DiscDate） | `equities/events.py:earnings_surprise, forecast_revision, dividend_forecast_revision, buyback_intensity` | §6.1 PEAD一次シグナル |
| 価値/質/サイズ・ファクタ | `equities/fundamental_factors.py` | §6.1 価値戦略一次シグナル |
| 実決算日（場中/引け後の別） | `data/processed/edinet_annual_index.parquet(DiscDate)`／`data/tdnet/`＋`data/sources/tdnet.py(disclosure_date/time, event_tags=earnings_release)` | **§2 実開示日＝循環フォールバック回避** |
| 戦略基底・単一資産タイミング | `research/strategy.py:Strategy, SignalTimingStrategy` | §6.2 レジーム適用 |
| 週足 walk-forward 検知器・配管・安定性 | `research/sector_regime/{detectors,pipeline,weekly,evaluation}.py` | §3 週足アンカー雛形・§5 日足検知器雛形・安定性ブートストラップ |

### 1.2 改修（小）

| 対象 | 現状 | 改修 |
|---|---|---|
| `sector_regime/detectors.py:_high_vol_state_index` | ボラ昇順固定 | `_align_states(states, key, ascending)` に一般化（**週足=モメンタム符号／日足=ボラ・Amihud**、アーキ3） |
| `sector_regime` 週足出力意味 | `regime_prob`=高ボラ確率 | **方向性アンカー生成器**を追加：S33代理上で bull/bear/neutral の filtered を出す（検知器機構は流用、整列・出力意味のみ変更） |
| `sector_regime/detectors.py:walk_forward_hmm` | GaussianHMM・diag 固定 | 放出（Gaussian/**Student-t**）・共分散（full/diag）・**sticky/min-dwell** をパラメータ化（§5/§7） |
| エンジン・コスト入力 | scalar 既定 | **Amihud パラメータ化の `costs_bps` DataFrame** を構築（係数は保守側固定）。エンジン本体は無改修（DF を受ける） |

### 1.3 新規（`daily_regime/` 内のグルー＋コア）

| モジュール | 役割 | 再利用先 |
|---|---|---|
| `tiers.py` | **Amihud の PIT 分位ティア（②）**を ADV ゲート（①）の上に。`TierMembership` PIT 系列を**遡及再付番なし**で出力 | universe（ゲート）・microstructure/price_factors（Amihud） |
| `features.py` | **median/MAD ロバストz＋構造成分除去（曜日/月内/SQ）＋Amihud フロア/対数＋チャネルタグ** | microstructure の生 Amihud を入力（ロバスト化・除去は新規） |
| `pooling.py` | **プール放出推定器**（ティア横断 ≤t を as-of-s ティアで積層）＋銘柄別 filter、λ固定 | detectors の HMM 機構を流用 |
| `multiscale.py` | **方式A（事前注入）/C（外生）＋ヒステリシス**。`WeeklyDirectionalAnchor` を asof 消費 | sector_regime 週足・AsOfView |
| `event_mask.py` | **決算チャネル分離**（リターン/ボラのみダミー化・流動性更新継続）、DiscDate＋場中/引け後で窓決定 | events.py/tdnet |
| `strategy.py` | `ValueStrategy`/`PEADStrategy`（`Strategy` 派生）＋レジーム filter/meta-label/sizing＋**§2適用除外** | strategy.py・events.py・fundamental_factors |
| `cost.py` | レジーム依存 `costs_bps` DF（Amihud・保守側固定）＋感応度グリッド | engine.backtest |
| `evaluate.py` | ベースライン群・**プラセボ（ブロック/位相）**・安定性スイープ（ティア境界/λ）・コスト感応度・DSR/CPCV 配線 | phase4_evaluate・dsr・cpcv_backtest |
| `leak_tests.py` | filtered vs smoothed ＋プラセボ・ハーネス | — |

---

## 2. シーム定義（データ契約・ソース束）

> §1-4 ラグ・PIT は**すべて `AsOf` 可用日**で表現。各契約は PIT 系列として出力し、シーム単体でリーク検査可能にする（filtered/smoothed 分離と同思想）。

### 2.1 週足filtered → 日足事前注入：`WeeklyDirectionalAnchor`（アーキ1×3の結合点）

| フィールド | 型 | 規約 |
|---|---|---|
| `week_end_date` (index) | date | 完了済み週の金曜（W-FRI） |
| `available_from` | date | アンカー可用日＝`week_end_date` の**翌営業日**（公表ラグ＝§1-4 の具体化） |
| `anchor_id` | str | 業種指数ID（**当面 S33 コード**＝`data_loader "project"`、実指数は将来） |
| `p_bear,p_neutral,p_bull` | float | filtered（Σ=1）。**モメンタム符号で整列**（アーキ3） |
| `regime` | enum | argmax ∈ {bear,neutral,bull} |

- **ソース束**：`sector_regime.data_loader("project")` → `build_weekly_features` → 方向性整列の週足検知器（1.2 改修）。
- **不変条件**：日足 t は `max(available_from ≤ t)` の1行のみ参照。`AsOf` 上では `frame("anchor_p_*")` が ≤t を保証、加えて `available_from = week_end_date+1営業日` で進行中週を排除。
- 擬似IF：`build_weekly_anchor(index_close, align="momentum") -> WeeklyDirectionalAnchor` ／ `inject_prior(base_prior, asof_row, mapping) -> daily_state_prior_t`（bear→日足 stressed 事前↑）。

### 2.2 流動性ティア PIT 所属：`TierMembership`（釘②）

| フィールド | 型 | 規約 |
|---|---|---|
| `(date, code)` (index) | — | 横断パネル |
| `tier` | enum | {`gate_illiquid`,`T1`,`T2`,`T3`}（ADVゲート＋Amihud3ティア） |
| `assigned_through` | date | 所属を決めた**過去分位**カットオフ（≤ date） |

- **ソース束**：①ゲート＝`universe.point_in_time_universe(turnover_panel)`／`liquid_universe_mask`（PIT）。②ティア＝`price_factors.amihud_illiq`（または `microstructure.amihud_illiquidity(close, turnover)`）の**PIT 過去分位ビニング（新規）**。
- **不変条件**：(1) 境界（ADV閾値・Amihud分位）は窓 `(t−W,t]` の過去のみ、refit ごと再計算。(2) **遡及再付番禁止**：refit τ のティア T プールは過去点 `(code,s≤τ)` を **as-of-s の `tier(code,s)`** で帰属（`tier(code,τ)` を遡及適用しない）。
- 擬似IF：`assign_tiers(panel_adv_amihud_leq_t, n_tiers=3, gate="adv_pit") -> TierMembership` ／ `fit_pooled_emissions(panel_feat_leq_t, membership_leq_t, tier, pooling_strength=λ_FIXED) -> EmissionParams` ／ `filter_name(name_feat_leq_t, pooled_emissions, name_transmat) -> filtered_probs_t`。

### 2.3 レジーム系列 → バックテスト

- 日足 filtered（`state_filtered`/`prob_stressed`）を **PIT 系列として `AsOfView` にフレーム登録** → `Strategy.target_weights(asof)` が `asof.frame("prob_stressed")` で ≤t 参照。`*_smoothed` は登録しない（§2.1 分離）。
- レジーム依存コスト：`cost.py` が Amihud（同時点）から `costs_bps` DF を構築 → `engine.backtest(costs_bps=DF)`。**係数 保守側固定＋感応度**（§6.4）。

### 2.4 決算チャネル分離（§2・§4.4）

- `event_mask.py` が DiscDate（events/tdnet）と**場中/引け後**から、(i) リターン/ボラ・チャネルのマスク日、(ii) PEAD エントリー窓（次バー起点）を生成。
- 出力もPIT系列。流動性チャネルは非マスク（更新継続）→ §6.3 サイジングは流動性更新後 filtered を読む。

---

## 3. 配置案

`invest_system/research/daily_regime/`（`sector_regime` の兄弟）。

```
research/daily_regime/
  __init__.py
  config.py        # tiers/λ/refit/窓・コスト係数（単一config）
  data.py          # store.load_wide + microstructure/price_factors + universe を束ねる薄い adapter
  tiers.py         # 新規：TierMembership（PIT・遡及再付番なし）
  features.py      # 新規：robust-z＋構造除去＋Amihud floor/log＋チャネルタグ
  pooling.py       # 新規：プール放出＋銘柄別filter（detectors流用）
  multiscale.py    # 新規：方式A/C＋ヒステリシス（sector_regime週足＋AsOfView）
  event_mask.py    # 新規：決算チャネル分離（events/tdnet）
  strategy.py      # 新規：Value/PEAD（Strategy派生）＋3経路＋§2除外
  cost.py          # 新規：レジーム依存costs_bps DF＋感応度
  evaluate.py      # 新規：ベースライン・プラセボ・安定性・DSR/CPCV配線
  leak_tests.py    # 新規：filtered/smoothed＋プラセボ
weekly_anchor.py（sector_regime 内 or daily_regime/multiscale 配下）  # 改修＋新規：方向性アンカー生成
```
横断再利用：`data/`、`equities/`、`features/`、`research/{engine,data_view,strategy}`、`validation/`、`backtest/`。

---

## 4. 広域指数フォールバック（(b)内・選択肢提示）

対象個別の S33 業種が薄い/代表性を欠く場合の方向性アンカー。**実 TOPIX/業種指数の組込ローダーは未確認**（`data/supplemental/` に `n225_iv` のみ）。

| 案 | 内容 | 評価 |
|---|---|---|
| A | S33 セクター代理のみ | 最小だが薄いセクターでノイズ |
| **B（推奨）** | S33 代理＋**PIT 構成銘柄数ゲート**：その週のセクター構成銘柄 < N のとき等加重マーケット代理（feature_store の `ret.mean(axis=1)`）へ切替 | **既存系列のみ・最も頑健・PIT 自然** |
| C | TOPIX-17 等の中間グルーピングを新規構築 | 表現力↑だがグルーピング設計＝過剰適合余地＋新規構築コスト |

→ **B 推奨**（N は §11 未決定の安定性確認項目に追加、最適化せず固定）。ここは確定でなく選択を仰ぐ。

---

## 5. 実装順序（設計書 §9・着手順序に整合）

0. **シーム＋PIT 配線を先に**（釘①）：`WeeklyDirectionalAnchor`／`TierMembership`／レジーム→`AsOf` 登録の契約を切り、`leak_tests.py`（filtered/smoothed＋ブロック/位相プラセボ）を**最初に**立てる。
1. **データ再利用配線**：`data.py` で load_wide＋microstructure＋price_factors＋universe を束ね、値幅/stale フラグ・チャネル区分まで（§4・新規は robust 化のみ）。
2. **単層を walk-forward で**：日足ボラ流動性HMM（まず銘柄別→次に pooling）と週足方向性アンカー（S33）。`engine.backtest`（再利用コスト/ターンオーバー）に通し、ベースライン「週足のみ／日足のみ」で配管を固める。
3. **モデル変種**：Gaussian vs t、full/diag、sticky/min-dwell/ヒステリシス、GMM 比較、**pooling 有無**。
4. **PEAD 適用除外**：`event_mask.py`（チャネル分離）＋ Value/PEAD `Strategy`＋§2 除外でバックテスト。
5. **マルチスケール**：方式A/C＋ヒステリシス＋セクター注入 on/off。
6. **総合**：全ベースライン・安定性スイープ（ティア境界/λ/開始日/銘柄）・**レジーム依存コスト感応度**・DSR/CPCV 多重検定・プラセボ → レポート。

---

## 6. 先読みの最弱点（プラセボで重点監視）
1. **ティア遡及再付番**（§2.2 不変条件2）：pooling 追加時に最も作り込みやすいリーク。as-of-s 帰属を契約で固定。
2. **週足アンカーのラグ**（§2.1 `available_from`）：進行中週の混入。翌営業日可用で排除。
3. **正規化・分位の窓**：robust-z／Amihud フロア／ティア境界の全期間統計化。`data_view`/過去窓で強制。
→ いずれも filtered/smoothed 乖離テストでは捉えにくいため、**ブロックシャッフル/位相ランダム化プラセボで edge≈0** を必須ゲートにする。
