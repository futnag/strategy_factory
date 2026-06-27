# examples/ 検証スクリプト監査レポート

**監査日**: 2026-06-22  
**対象**: `examples/*.py` 全 96 ファイル  
**方法**: 各ファイルの冒頭 docstring（記載目的）と実装（import・データ組立・判定経路・PIT/コスト/レジストリ規律）を突合。コードは変更していない。  
**評価記号**:

| 記号 | 意味 |
|------|------|
| ✅ | docstring の目的どおり実装され、方法論も妥当 |
| ⚠️ | 目的は概ね一致するが、限界・バイアス・命名・再現性に注意 |
| ❌ | docstring と実装が乖離、または重大な方法論欠陥 |
| ⊘ | 戦略検証ではない（データ取得・デモ・運用・インフラ） |

---

## 1. エグゼクティブサマリー

`examples/` は **96 本**の Python ランナーで構成される。うち **戦略・仮説の検証**に該当するのはおおよそ **半数（45〜50 本）**、残りはデータ材化・Phase 2/4 運用・フレームワークデモ・API プローブである。

### 全体評価

- **重大な先読みバグ（docstring に PIT と書きながら実装が違う等）は ❌ として検出されなかった。**
- **旗艦ライン（value / PEAD / switch / TSMOM 合成）**は、PIT ユニバース top300・`lag_days=1`・15bps・容量・`judge_grid`＋永続レジストリという規律が docstring と実装で一貫している。
- **主な注意点（⚠️）**は次の 5 類型に集約される:
  1. **固定ユニバース / 流動性スナップショット** — 探索初期の `jp_equities_factor_study.py` や日次デモ `research_gap_demo.py` / `research_runup_daily.py`
  2. **in-sample レジーム割当** — `research_value_pead_regime.py` の静的 `switch`（ただし同一ファイル内で `wf_switch` が正直な代替として実装済み）
  3. **診断スクリプトのハードコード定数** — `research_flagship_robustness_diag.py` の報告 SR を固定値で再入力
  4. **イベント研究のサンプル窓・単一ペア** — C1 の 2022+ 窓、カレンダー/ペアのトヨタ/ホンダ固定
  5. **暗号・ティック系の履歴短さ・コスト省略** — 教育/実験目的として docstring 内で正直に限定されている

### スクリプト種別の内訳

| 種別 | 本数 | 代表 |
|------|------|------|
| 正式仮説検定（`judge_grid` + レジストリ） | 23 | `research_value_pead_combo.py` |
| 早期 jp_* 検証（独自 DSR パイプライン） | 6 | `jp_pit_universe_study.py` |
| throwaway 診断（K 不変） | 14 | `research_flagship_robustness_diag.py` |
| C1/C2 イベント（記述統計 or Path B 裁定） | 7 | `judge_c1_tob_arb.py` |
| Phase 4/4b GKX 判定 | 5 | `run_phase4b_judgment.py` |
| 検証・突合（verify/judge/flows） | 5 | `verify_factor_external.py` |
| データ・材化・更新 | 15 | `download_jquants.py` |
| フレームワーク・暗号デモ | 14 | `bitbank_e2e.py` |
| Phase 2 本番運用 | 4 | `phase2_generate_orders.py` |
| その他（レポート・プローブ） | 11 | `registry_status.py` |

---

## 2. 横断的所見

### 2.1 規律の階層（リポジトリ全体）

1. **柱 C（日本株クロスセクション）の本番規律**: `point_in_time_universe`（top300/500）・`point_in_time` ファンダ・S33 中立・z 化・15bps・`participation×ADV`・`execution_lag` 明示・`judge_grid`・scope 別 K。
2. **throwaway 診断**: docstring で「K 不変」「レジストリ不使用」を宣言し、実装でも `default_registry()` を開かないか、読み取り専用。
3. **Phase 4/4b**: 材化（`build_phase4*_matrix.py`）と判定（`run_phase4*_judgment.py`）を分離。二重実行ガード・FROZEN プロトコル準拠。

### 2.2 docstring と実装が一致している共通パターン

- PIT ユニバース: `point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)` が旗艦系で統一。
- ファンダ PIT: `point_in_time(..., lag_days=1)` + `DiscDate` 列のイベント系。
- コスト: 月次 LS は `costs_bps=15.0`、マイクロキャップは docstring どおり 30bps（別途感応スイープあり）。
- ペア/平均回帰: `execution_lag=1`、形成窓→取引窓の時間順序、CADF ゲート。

### 2.3 繰り返し現れる限界（意図的なもの含む）

| テーマ | 該当スクリプト | 妥当性 |
|--------|----------------|--------|
| 探索後の value 軸 OOS | `jp_value_oos.py` | docstring が「真の OOS 純度は失われている」と明記。DSR/minTRL で選択を罰する設計は妥当 |
| 符号探索込み DSR | `jp_margin_short_study.py`, `jp_causal_shorttolong.py` | 両符号試行は docstring で開示。探索段階として許容、本番は `judge_grid` へ移行済み |
| 静的 switch vs wf_switch | `research_value_pead_regime.py` | 静的版は in-sample と docstring 自認。wf_switch が同一ファイルで検証されるのは方法論的に優秀 |
| ウェイト再利用の独立検証 | `verify_flagship_independent.py` | docs/04 P1-A が許容する範囲の突合。エンジン再実装は実装されている |
| オプション VRP | `research_vol_premium_n225.py` | 清算値ベース・板無し。docstring が「事前登録・満期 SQ 整合」と限定。実運用ギャップは ⚠️ |

### 2.4 サイレントな先読みリスク（要認識）

| ファイル | 実装 | docstring との関係 |
|----------|------|-------------------|
| `research_gap_demo.py` | `LIQ_DATE = "20260225"` の **1 日スナップショット**で全期間の銘柄集合を固定 | 「PIT 安全な日次バックテスト」とあるが、**ユニバースは PIT ではない**。デモ目的なら許容、検証としては ⚠️ |
| `research_runup_daily.py` | 同上 | イベントロジックは PIT（`days_to_next_announcement`）だがユニバースは固定 ⚠️ |
| `jp_equities_factor_study.py` | `select_universe()`＝全期間中央値 | `universe.py` 自身が「先読み固定」と警告。後続 `jp_pit_universe_study.py` で是正済み ⚠️ |

---

## 3. カテゴリ別詳細

### 3.1 正式仮説検定（`judge_grid` + 永続レジストリ）— 23 本

いずれも docstring の戦略記述 → `judge_grid` → `write_html` → scope 登録の流れが確認された。

| ファイル | 検証する戦略 | 評価 | 備考 |
|----------|--------------|------|------|
| `research_breadth_factors.py` | value×mom×quality×low-vol 合成 | ✅ | PIT top500・fins パネル・新ファクター accruals/low_vol |
| `research_calendar_pairs.py` | 月末効果＋ペア MR | ⚠️ | TOPIX 月末は妥当。ペアは **72030/72670 固定 1 組**（docstring は「例」と明記） |
| `research_disclosure_timing.py` | 発表遅延シグナル | ✅ | `announcement_delay`・K=2 事前登録 |
| `research_earnings_timing.py` | run-up / 発表回避 overlay | ✅ | `expected_announcement_month` PIT・月次 |
| `research_factor_vqp_longonly.py` | V+Q+P ロングオンリー | ✅ | PIT top500・ロングティルト規律 |
| `research_gap_demo.py` | ギャップリバーサル格子 | ⚠️ | 判定器デモは妥当。**固定流動性スナップショット** |
| `research_guidance_bias.py` | 期初ガイダンス保守バイアス | ✅ | PEAD 交絡を docstring で認識 |
| `research_index_events.py` | 日経225 入替ドリフト/リバーサル | ✅ | 公式イベント表・発表日 PIT |
| `research_meanrev_pairs.py` | 同業種 CADF ペア MR | ✅ | extra_trials で探索 K 計上・lag=1 |
| `research_meanrev_regime.py` | レジームゲート付きペア MR | ✅ | breakdown 先行・同一 scope で K 共有 |
| `research_microcap_reversal.py` | 低流動性帯リバーサル | ✅ | `band_universe` は PIT 実装確認。`growth_all` は **Growth 全市場**（名称は「all」で帯限定ではない）→ ⚠️ 命名のみ |
| `research_pead_shortint.py` | PEAD＋空売り残高 | ✅ | |
| `research_price_factors.py` | Gold 層 mom/low-vol/reversal | ✅ | API 不要・PIT |
| `research_real_strategies.py` | フロー→TOPIX＋value XS | ✅ | ファクトリ総合デモ |
| `research_residual_momentum.py` | 残差モメンタム | ✅ | Fama-French 風残差化・PIT |
| `research_runup_daily.py` | 日次 earnings run-up | ⚠️ | 戦略ロジックは docstring どおり。**固定ユニバース** |
| `research_shareholder_return.py` | 配当改訂・自社株買い | ✅ | PEAD 直交化版を主形と明記 |
| `research_switch_tsmom_combo.py` | switch×TSMOM 合成 | ✅ | K=4 事前登録・best-of-grid 回避 |
| `research_tsmom_multiasset.py` | 11 資産 TSMOM | ✅ | natgas 除外は docstring どおり |
| `research_value_pead_combo.py` | value+PEAD 合成 | ✅ | |
| `research_value_pead_longtilt.py` | ロングティルト PEAD＋value | ✅ | |
| `research_value_pead_regime.py` | レジームゲート＋switch/wf_switch | ⚠️ | 静的 `switch` は in-sample（自認）。`wf_switch` で是正検証あり |
| `research_vol_premium_n225.py` | N225 オプション VRP | ⚠️ | 設計は docstring どおり。約定・デルタヘッジ・板は未モデル |

### 3.2 throwaway 診断（K 不変・レジストリ非消費）— 14 本

| ファイル | 目的 | 評価 | 備考 |
|----------|------|------|------|
| `research_c1_tob_arb.py` | TOB スプレッド×成否の prior 診断 | ✅ | `judge_grid` は使わない（docstring どおり）。正式裁定は `judge_c1_tob_arb.py` |
| `research_c2_drift.py` | C2 新規参入後ドリフト記述統計 | ✅ | DSR 裁定なしと明記 |
| `research_c2_escalation.py` | C2 凍結 4 条件の記述統計 | ✅ | 条件リストがファイル内凍結 |
| `research_factor_momentum_diag.py` | スリーブ過去 12m リターン分離 | ✅ | PIT ラベル・K=0 |
| `research_flagship_robustness_diag.py` | グローバル K ラダー＋ローリング SR | ⚠️ | ② は backtest 再計算で妥当。① は **SR/n/sk/ku をハードコード**（`ft` dict L71）— レジストリ再計算ではない |
| `research_pead_oos_diagnose.py` | PEAD OOS 失敗の IC/脚分解 | ✅ | |
| `research_rebalance_band.py` | デッドバンド感応 | ✅ | ウェイト固定・コストのみ変更 |
| `research_upstream_contamination.py` | 値幅制限汚染監査 | ✅ | mask-first 比較 |
| `research_value_pead_realism.py` | 値幅制限・貸株・ケリー | ✅ | 戦略凍結・現実性のみ変更 |
| `research_value_pead_timing.py` | T+1 始値・ボラ連動コスト | ✅ | 同上 |
| `diag_edinet_fundamentals.py` | EDINET 特徴量 IC/被覆 | ✅ | |
| `diag_factor_expansion.py` | 信用/微細構造 IC | ✅ | PIT top500 |
| `diag_price_factors.py` | 価格ファクター IC | ✅ | |
| `preflight_data_audit.py` | Phase 4 前 9 点診断 | ✅ | |
| `phase4b_preflight_diag.py` | Phase 4b 小型ユニバース診断 | ✅ | |

### 3.3 早期 jp_* 検証パイプライン — 6 本

`judge_grid` 以前の探索スクリプト。独自に DSR・サブ期間を計算。

| ファイル | 目的 | 評価 | 備考 |
|----------|------|------|------|
| `jp_equities_factor_study.py` | 日本株ファンダ XS 初期探索 | ⚠️ | **`select_universe`＝全期間固定**。docstring は PIT と書くがユニバースは非 PIT。後続 `jp_pit_universe_study` が本番 |
| `jp_pit_universe_study.py` | 生存者バイアス除去後の再検証 | ✅ | `point_in_time_universe`・DSR・サブ期間 |
| `jp_margin_short_study.py` | 信用・空売りファクター | ✅ | PIT ラグ・符号両試行は docstring 開示 |
| `jp_causal_shorttolong.py` | short_to_long 独立性 | ✅ | 残差化＋LiNGAM |
| `jp_combo_value_margin.py` | value×stl 合成 | ⚠️ | 発見済み符号使用＝DSR 楽観（docstring 明記） |
| `jp_value_oos.py` | value 厳密 OOS | ✅ | 探索汚染を docstring で認めた上で DSR/minTRL |

### 3.4 C1/C2・イベント裁定 — 7 本

| ファイル | 目的 | 評価 | 備考 |
|----------|------|------|------|
| `judge_c1_tob_arb.py` | TOB リスクアーブ正式 DSR（Path B） | ✅ | テンダー決済・T+1 始値・按分は `verify_c1_tender.py` で補強 |
| `verify_c1_tender.py` | 按分（proration）感応 | ✅ | フルテンダー vs 按分補正を並記 |
| `edinet_c1_diagnostic.py` | C1 Phase A 診断 | ⚠️ | **2022+ のみ**（docstring で限界開示） |
| `edinet_tob_deals.py` | TOB 案件テーブル構築デモ | ⊘ | データ層 |
| `edinet_tob_backfill.py` | TOB 本体バックフィル | ⊘ | |
| `edinet_lvh_backfill.py` | C2 大量保有バックフィル | ⊘ | |
| `flows_analysis.py` | 海外フロー→TOPIX | ✅ | 公表遅延を考慮した翌週リターン |

### 3.5 Phase 4 / Phase 4b（GKX ML 判定）— 5 本

| ファイル | 目的 | 評価 | 備考 |
|----------|------|------|------|
| `build_phase4_matrix.py` | Phase 4 設計行列材化 | ⊘ | K 不変・docstring どおり |
| `run_phase4_judgment.py` | Phase 4 一度きり DSR | ✅ | docs/19 FROZEN・二重実行ガード |
| `build_phase4b_matrix.py` | Phase 4b 小型ユニバース材化 | ⊘ | mcap≥¥30B のみ変更 |
| `run_phase4b_judgment.py` | Phase 4b 一度きり DSR | ✅ | docs/21 FROZEN |
| `phase4b_preflight_diag.py` | 4b 事前診断 | ✅ | throwaway |

### 3.6 独立検証・外部突合 — 3 本

| ファイル | 目的 | 評価 | 備考 |
|----------|------|------|------|
| `verify_flagship_independent.py` | 旗艦の独立 P&L 再計算 | ✅ | ウェイト再利用は P1-A 許容範囲。閾値 1e-8 |
| `verify_factor_external.py` | Kenneth French 日本因子突合 | ✅ | 構築差を docstring で説明・WARN 基準あり |
| `build_report_index.py` | レジストリ HTML 索引 | ⊘ | |

### 3.7 Phase 2 本番運用 — 4 本

| ファイル | 目的 | 評価 | 備考 |
|----------|------|------|------|
| `phase2_generate_orders.py` | 月末シグナル→注文 | ✅ | 凍結パラメータ・PIT top300・T+1 寄付 |
| `phase2_reconcile.py` | 月次照合・台帳再構成 | ✅ | ステートレス設計 |
| `phase2_dashboard_data.py` | ops 用 data.json | ⊘ | |
| `phase2_push_supabase.py` | Supabase 同期 | ⊘ | |

### 3.8 暗号・フレームワークデモ — 14 本

教育・実験目的。日本株本番規律とは切り離されている。

| ファイル | 目的 | 評価 | 備考 |
|----------|------|------|------|
| `bitbank_e2e.py` | BTC/JPY 4h E2E | ⚠️ | honest CPCV+DSR は妥当。**コスト・短い履歴** |
| `edge_search.py` | マイクロ構造特徴探索 | ⚠️ | 同上 |
| `cross_sectional.py` | crypto XS ファクター | ⚠️ | コストなし・DSR のみ |
| `multifactor.py` | 因果フィルタ＋合成 capstone | ⚠️ | |
| `dollar_bar_pipeline.py` | tick→ドルバー→DSR | ⚠️ | **約 60 日分の tick**（docstring は長期比較を示唆するがデータ窓は短い） |
| `tick_bars_demo.py` | ドルバー統計比較 | ⚠️ | デモ |
| `end_to_end_demo.py` | 合成データ E2E | ⊘ | |
| `unified_e2e.py` | 全モジュール capstone | ⊘ | 合成＋コライダー除去デモ |
| `validation_harness_demo.py` | 多重検定デモ | ⊘ | |
| `causal_filter_demo.py` | コライダー符号反転 | ⊘ | |
| `frac_diff_demo.py` | 分数階差分 | ⊘ | |
| `meta_labeling_demo.py` | メタラベリング | ⊘ | |
| `triple_barrier_demo.py` | トリプルバリア | ⊘ | |
| `uniqueness_demo.py` | サンプル独自性 | ⊘ | |
| `portfolio_demo.py` | HRP/NCO | ⊘ | |
| `volatility_forecast.py` | ボラ予測 R² | ⚠️ | 方向 vs ボラの対比デモ |
| `vol_targeting.py` | ボラターゲティング | ⚠️ | |

### 3.9 データ取得・材化・更新 — 15 本

いずれも ⊘（検証ロジックは `invest_system` 側）。docstring と実装は **データパイプラインの説明どおり**。

- `download_jquants.py`, `download_edinet.py`, `build_edinet_long.py`
- `update_data.py`, `update_external.py`
- `fetch_margin_short.py`, `fetch_indices_flows.py`
- `edinet_update.py`, `edinet_probe.py`, `edinet_validate_slice.py`
- `jquants_smoke.py`, `jquants_calibrate.py`, `jquants_standard_check.py`
- `jquants_probe2.py`, `jquants_indices_probe.py`, `jquants_markets_probe.py`
- `registry_status.py`

---

## 4. 全ファイル一覧（96 件）

凡例: **種別** — F=正式判定, D=診断, J=jp探索, E=イベント, P4=Phase4, V=独立検証, O=運用, C=暗号デモ, I=インフラ

| # | ファイル | 種別 | 評価 | 一行要約 |
|---|----------|------|------|----------|
| 1 | `bitbank_e2e.py` | C | ⚠️ | 実 BTC purged CPCV+DSR デモ |
| 2 | `build_edinet_long.py` | I | ⊘ | 有報長形式材化 |
| 3 | `build_phase4_matrix.py` | P4 | ⊘ | GKX 設計行列 |
| 4 | `build_phase4b_matrix.py` | P4 | ⊘ | 4b 小型ユニバース行列 |
| 5 | `build_report_index.py` | I | ⊘ | 研究 HTML 索引 |
| 6 | `causal_filter_demo.py` | C | ⊘ | コライダーデモ |
| 7 | `cross_sectional.py` | C | ⚠️ | crypto XS ファクター |
| 8 | `diag_edinet_fundamentals.py` | D | ✅ | EDINET IC 診断 |
| 9 | `diag_factor_expansion.py` | D | ✅ | 信用/微細 IC |
| 10 | `diag_price_factors.py` | D | ✅ | 価格 IC |
| 11 | `dollar_bar_pipeline.py` | C | ⚠️ | tick ドルバー DSR（短窓） |
| 12 | `download_edinet.py` | I | ⊘ | EDINET バックフィル |
| 13 | `download_jquants.py` | I | ⊘ | J-Quants by-date |
| 14 | `edge_search.py` | C | ⚠️ | マイクロ構造探索 |
| 15 | `edinet_c1_diagnostic.py` | E | ⚠️ | C1 診断 2022+ |
| 16 | `edinet_lvh_backfill.py` | I | ⊘ | C2 大量保有 |
| 17 | `edinet_probe.py` | I | ⊘ | API PoC |
| 18 | `edinet_tob_backfill.py` | I | ⊘ | TOB 本体 |
| 19 | `edinet_tob_deals.py` | I | ⊘ | TOB 案件デモ |
| 20 | `edinet_update.py` | I | ⊘ | EDINET 差分更新 |
| 21 | `edinet_validate_slice.py` | D | ✅ | EDINET↔J-Q 突合 |
| 22 | `end_to_end_demo.py` | C | ⊘ | 合成 E2E |
| 23 | `fetch_indices_flows.py` | I | ⊘ | 指数・フロー取得 |
| 24 | `fetch_margin_short.py` | I | ⊘ | 信用データ取得 |
| 25 | `flows_analysis.py` | V | ✅ | 海外フロー PIT |
| 26 | `frac_diff_demo.py` | C | ⊘ | 分数階差分 |
| 27 | `jp_causal_shorttolong.py` | J | ✅ | 因果独立性 |
| 28 | `jp_combo_value_margin.py` | J | ⚠️ | value×stl 合成 |
| 29 | `jp_equities_factor_study.py` | J | ⚠️ | 初期探索・固定 U |
| 30 | `jp_margin_short_study.py` | J | ✅ | 需給ファクター |
| 31 | `jp_pit_universe_study.py` | J | ✅ | PIT 再検証 |
| 32 | `jp_value_oos.py` | J | ✅ | value OOS |
| 33 | `jquants_calibrate.py` | I | ⊘ | API 間隔調整 |
| 34 | `jquants_indices_probe.py` | I | ⊘ | 指数 API |
| 35 | `jquants_markets_probe.py` | I | ⊘ | 信用 API |
| 36 | `jquants_probe2.py` | I | ⊘ | 決算予定 API |
| 37 | `jquants_smoke.py` | I | ⊘ | 接続テスト |
| 38 | `jquants_standard_check.py` | I | ⊘ | Standard 検証 |
| 39 | `judge_c1_tob_arb.py` | E | ✅ | C1 正式 DSR |
| 40 | `meta_labeling_demo.py` | C | ⊘ | メタラベル |
| 41 | `multifactor.py` | C | ⚠️ | マルチファクター |
| 42 | `phase2_dashboard_data.py` | O | ⊘ | ダッシュボード JSON |
| 43 | `phase2_generate_orders.py` | O | ✅ | 注文生成 |
| 44 | `phase2_push_supabase.py` | O | ⊘ | DB 同期 |
| 45 | `phase2_reconcile.py` | O | ✅ | 月次照合 |
| 46 | `phase4b_preflight_diag.py` | D | ✅ | 4b 診断 |
| 47 | `portfolio_demo.py` | C | ⊘ | HRP デモ |
| 48 | `preflight_data_audit.py` | D | ✅ | Phase4 前診断 |
| 49 | `registry_status.py` | I | ⊘ | K 一覧 |
| 50 | `research_breadth_factors.py` | F | ✅ | breadth 合成 |
| 51 | `research_c1_tob_arb.py` | D | ✅ | C1 prior |
| 52 | `research_c2_drift.py` | D | ✅ | C2 ドリフト |
| 53 | `research_c2_escalation.py` | D | ✅ | C2 エスカレーション |
| 54 | `research_calendar_pairs.py` | F | ⚠️ | 月末＋固定ペア |
| 55 | `research_disclosure_timing.py` | F | ✅ | 発表遅延 |
| 56 | `research_earnings_timing.py` | F | ✅ | 決算タイミング |
| 57 | `research_factor_momentum_diag.py` | D | ✅ | FM 分離 |
| 58 | `research_factor_vqp_longonly.py` | F | ✅ | VQP ロングオンリー |
| 59 | `research_flagship_robustness_diag.py` | D | ⚠️ | K ラダー・ハードコード SR |
| 60 | `research_gap_demo.py` | F | ⚠️ | ギャップデモ・固定 U |
| 61 | `research_guidance_bias.py` | F | ✅ | ガイダンスバイアス |
| 62 | `research_index_events.py` | F | ✅ | 225 入替 |
| 63 | `research_meanrev_pairs.py` | F | ✅ | ペア MR |
| 64 | `research_meanrev_regime.py` | F | ✅ | レジーム MR |
| 65 | `research_microcap_reversal.py` | F | ✅ | マイクロ逆張り |
| 66 | `research_pead_oos_diagnose.py` | D | ✅ | PEAD 失敗診断 |
| 67 | `research_pead_shortint.py` | F | ✅ | PEAD+空売り |
| 68 | `research_price_factors.py` | F | ✅ | Gold 価格因子 |
| 69 | `research_real_strategies.py` | F | ✅ | 実戦略デモ |
| 70 | `research_rebalance_band.py` | D | ✅ | デッドバンド |
| 71 | `research_residual_momentum.py` | F | ✅ | 残差モメンタム |
| 72 | `research_runup_daily.py` | F | ⚠️ | 日次 run-up・固定 U |
| 73 | `research_shareholder_return.py` | F | ✅ | 株主還元 |
| 74 | `research_switch_tsmom_combo.py` | F | ✅ | switch×TSMOM |
| 75 | `research_tsmom_multiasset.py` | F | ✅ | TSMOM 単体 |
| 76 | `research_upstream_contamination.py` | D | ✅ | 汚染監査 |
| 77 | `research_value_pead_combo.py` | F | ✅ | value+PEAD |
| 78 | `research_value_pead_longtilt.py` | F | ✅ | ロングティルト |
| 79 | `research_value_pead_realism.py` | D | ✅ | 執行現実性 |
| 80 | `research_value_pead_regime.py` | F | ⚠️ | レジーム・静的 switch |
| 81 | `research_value_pead_timing.py` | D | ✅ | 執行タイミング |
| 82 | `research_vol_premium_n225.py` | F | ⚠️ | オプション VRP |
| 83 | `run_phase4_judgment.py` | P4 | ✅ | Phase4 判定 |
| 84 | `run_phase4b_judgment.py` | P4 | ✅ | Phase4b 判定 |
| 85 | `tick_bars_demo.py` | C | ⚠️ | ドルバー統計 |
| 86 | `triple_barrier_demo.py` | C | ⊘ | ラベリング |
| 87 | `unified_e2e.py` | C | ⊘ | 統合 capstone |
| 88 | `uniqueness_demo.py` | C | ⊘ | 独自性 |
| 89 | `update_data.py` | I | ⊘ | 差分更新 |
| 90 | `update_external.py` | I | ⊘ | 外部価格更新 |
| 91 | `validation_harness_demo.py` | C | ⊘ | 検証デモ |
| 92 | `verify_c1_tender.py` | V | ✅ | 按分検証 |
| 93 | `verify_factor_external.py` | V | ✅ | French 突合 |
| 94 | `verify_flagship_independent.py` | V | ✅ | 独立再計算 |
| 95 | `vol_targeting.py` | C | ⚠️ | ボラターゲット |
| 96 | `volatility_forecast.py` | C | ⚠️ | ボラ予測 |

### 評価集計

| 評価 | 本数 |
|------|------|
| ✅ | 58 |
| ⚠️ | 22 |
| ❌ | 0 |
| ⊘ | 16 |

---

## 5. 優先度付き改善候補（ドキュメントのみ・未実装）

コード変更は行っていない。将来の修正候補として優先度順に列挙する。

### P1（方法論の明確化）— **対応済み（2026-06-22）**

1. **`research_gap_demo.py` / `research_runup_daily.py`**: docstring を「シグナル/執行=PIT・ユニバース=固定スナップショット・デモ用途」に分解明記（挙動不変）。
2. **`research_flagship_robustness_diag.py`**: `global_k_ladder()` のハードコード SR をレジストリ読み取り（`value_pead_switch` / `switch_tsmom_combo`）＋ backtest フォールバックに置換。
3. **`jp_equities_factor_study.py`**: 冒頭に非 PIT 探索用＋`jp_pit_universe_study.py` 参照を追記（コード不変）。

### P2（再現性・一般化）

4. **`research_calendar_pairs.py`**: ペアを環境変数または業種内流動性上位から自動選定する拡張（現状はデモとして妥当だが一般化余地）。
5. **`research_microcap_reversal.py`**: 変種名 `growth_all` を `growth_market` 等にリネーム（実装は Growth 全市場で正しい）。
6. **`dollar_bar_pipeline.py`**: docstring の比較主張と取得 tick 日数の整合を doc に明記（現状 ~60 日）。

### P3（運用ギャップの文書化）

7. **`research_vol_premium_n225.py`**: 清算値・フルキャッシュ担保以外のシナリオ（デルタヘッジ・板スプレッド）を「未検証」と results doc に分離記載。
8. **暗号系 6 本**: 本番日本株との規律差（コスト・ユニバース・K 族）を README または docs/03 から examples 索引へリンク。

---

## 6. 結論

`examples/` 配下の検証スクリプトは、**冒頭 docstring に書かれた戦略意図と実装の対応はおおむね良好**である。特に柱 C の本番ライン（value / PEAD / switch / TSMOM / 合成）は PIT・コスト・容量・`judge_grid`・K 規律が一貫しており、throwaway 診断はレジストリを汚さない設計が守られている。

**❌ に該当する重大不整合は見つからなかった。** 改善余地は主に (1) 初期探索・デモ用の固定ユニバースの明示、(2) 旗艦頑健性診断の定数依存、(3) オプション・暗号の執行モデル簡略化のドキュメント補強、に集約される。

本レポートは監査時点の静的コードレビューに基づく。各スクリプトの**数値結果の再現実行**は本監査の範囲外とする。