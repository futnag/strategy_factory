# 19. Phase 4 事前登録メモ（GKX 型 ML・一度きりの DSR 判定）

> **STATUS: FROZEN（2026-06-21・ユーザー承認済み）— OOS 結果を見る前に本設定を凍結した。
> `examples/run_phase4_judgment.py` を一度だけ実行する。以後、通すための事後変更（特徴量追加・
> ユニバース変更・再チューニング・再判定）は一切しない（handoff §0,§3,§6,§10）。**
> 作成 2026-06-21。関連：docs/14（EDINET）・docs/16（プリフライト）・docs/17（設計判断）・docs/18（符号/重複）。

この判定は**貴重な一回**。リポジトリ全体の反 p-hacking 機構（CPCV・デフレート DSR・永続レジストリ・
purge/embargo）はこの瞬間のためにある。**結果が芽が出なくても受け入れる**。通すための事後変更（特徴量
追加・ユニバース変更・再チューニング・再判定）は一切しない（handoff §0,§3）。

---

## 1. 凍結された入力（docs/17・§2 確定済み）

- **ユニバース**：本番＝流動（株価≥¥100・時価総額≥¥10B・60日ADV≥¥5000万・普通株・PIT）。
  `design_matrix.production_universe`（=`universe.liquid_universe_mask`）。月平均 ~1,880 銘柄。
- **リターン/ラベル**：`y[t,code]` = 1M 先・**トータル**（配当込み）・**超過**（本番ユニバース断面等加重
  平均控除）・close[t]→close[t+1]・**廃止 last-price 補完**。`design_matrix.build_label`。
- **特徴量（凍結・新規作成なし）**：60 列（重複解消済み）。
  - 価格/流動性 15、微細構造 6、信用/空売り 6、ファンダ拡張B 12（materialized・月末 PIT）。
  - value/quality/size 10（`value_quality_size_factors`）。
  - EDINET 三表 11（`edinet_factor_panels`・**分割調整済 net_share_issuance** 含む）。
  - **重複解消（docs/18）**：accruals は EDINET 版のみ（JQ 版は符号逆＝除外）／CF/P は `cf_to_price` 1 本
    （`cf_yield` 除外）／12-1 モメンタムは `momentum` 1 本。
- **標準化（毎月クロスセクション・一元化）**：本番ユニバース内で
  `winsorize_cross_sectional(1/99%)` → `rank_and_fill([-1,1]・欠損中立0)` → （未使用：sector_neutralize
  は線形解釈用にオプション）。
- **PIT**：materialized は月末スナップショット、value/EDINET は DiscDate as-of（lag 1 営業日）、
  macro は公表ラグ。結合パネルの先読み不変は test_design_matrix／test_equities で green。

## 2. マクロ状態変数（特性 × マクロ交互作用）

PIT 整合の状態変数（`design_matrix.macro_state`・`external.asof_align` で公表ラグ）：

| 状態変数 | 定義/出所 | ラグ規約 |
|---|---|---|
| `jp_10y` | JGB 10年（macro_extended） | 統計系→ **1ヶ月ラグ**相当（monthly ffill＋lag 1 営業日） |
| `term_spread` | `jp_10y − jp_policy` | 同上 |
| `vix` | VIX（日次） | ラグ 1 営業日 |
| `n225_iv` | 日経225 IV（options_225 の ATM IV を日次集約・`data/supplemental/n225_iv.parquet`） | 取引日=公表、ラグ 1 |
| `foreign_flow` | 海外投資家ネット強度（投資部門別 `FrgnBal/FrgnTot`・`PubDate` アンカー） | 公表日アンカー |

- **線形モデル**：特性 × マクロの **Kronecker 交互作用列**を明示生成（`add_interactions`）。
  60 特性 × 5 マクロ = 300 交互作用 ＋ 60 特性 ＋ 5 マクロ = **365 列**。Lasso/ElasticNet で正則化。
- **木/NN**：生の特性＋マクロ（broadcast）のみ＝交互作用はモデルが学習（次元爆発回避・handoff §4）。
- **credit spread は全データに存在せず除外**（新規ソース追加は Phase 4 スコープ外＝handoff §1）。
  系列別ラグ規約は本表のとおり docs に明記済み。

## 3. 推定器スイート（`research/phase4_estimators.py`・seed=20260621・決定的）

| 族 | モデル | ハイパラ決定（**内側 CV/train 内のみ**） |
|---|---|---|
| 線形 | OLS（LinearRegression） | なし（基準） |
| 線形 | Lasso（LassoCV） | α を train 内 5-fold CV |
| 線形 | ElasticNet（ElasticNetCV） | α・l1_ratio∈{.1,.5,.9} を train 内 5-fold CV |
| 木 | RandomForest | 固定（n_estimators=200, max_depth=8, min_leaf=200, sqrt） |
| 木 | HistGradientBoosting | early stopping（内部 validation 15%） |
| NN | MLP（32,8・StandardScaler 同梱） | early stopping（内部 validation 15%） |

- **ベースライン先行**：OLS/Lasso/ElasticNet を確立し、**ML の増分を線形比で必ず併記**（handoff §3-2,§5）。
- **チューニングは検証のみ**：テスト/OOS フォールドでチューニングしない（*CV と early stopping は train 内）。
- 入力は族別（線形=交互作用込み 365 列／木・NN=生 65 列）。`make_inputs`。

## 4. 検証プロトコル（リーク無し OOS）

- **判定 OOS**：**月グループ Combinatorial Purged CV（CPCV）**（`cpcv_paths_predict`・月軸）。
  - `n_splits = 6`、`n_test_splits = 2` → C(6,2)=15 分割・**φ = C(5,1) = 5 パス/族**。
  - 各族で 5 つの完全 OOS パスを再構成し、**パスごとの L/S ネット Sharpe** を得る＝**Sharpe を分布で**評価。
  - **purge/embargo**：各テスト月の前 `gap_before=1ヶ月`（ラベル 1M フォワードの重複）を purge、後 `gap_after`
    ヶ月（**最長特徴量窓を跨ぐ後ろ向きリークの遮断**・§7 確認事項）を embargo。train obs の情報窓
    `[m−特徴量窓, m+1]` がテストと重なる場合に落とす（label＋feature 両側）。
- **開発診断（throwaway・別経路）**：拡張窓（min_train=36ヶ月・年次リフィット）で GKX 流 OOS R² と
  線形 vs ML 増分も併記可（厳密に因果）。本判定の R²/IC は CPCV パスの consensus 予測から算出。
- **OOS 判定期間**：被覆が安定する **2018-01 以降**を判定対象（学習は全期間・判定の主対象を 2018+）。
  学習窓・欠損方針は事前固定。

## 5. 判定指標・選択規則・デフレーション（**一度きり**）

1. **指標**：本番ユニバース・**手数料後**（片道 15bps・月次回転率連動）・**廃止 last-price 補完**の
   **上位/下位デシル L/S ポートフォリオ**の、**CPCV-OOS（φ=5 パス）月次ネットリターンの per-period
   Sharpe**（`phase4_evaluate.ls_portfolio_returns` → `dsr.sharpe_ratio`）。
2. **選択規則（事前宣言）**：6 族を各 1 回（固定設定）走らせ、**φ=5 パスの平均ネット Sharpe が最大の族を採用**。
3. **デフレーション（一度きり）**：`dsr.deflated_sharpe_ratio(sr=採用族の平均パス Sharpe,
   sr_variance=V[採用族の φ=5 パス間 Sharpe], n_trials=6, n_obs/skew/kurt=consensus 月次ネット, …)`。
   ＝**V[SR] は CPCV パス間分散**（推定ノイズ）、**N=6 は探索した族数**で E[max SR] を膨張補正
   （False Strategy 定理）。**φ（パス数）は試行数 N に算入しない**（パスは同一戦略の再標本）。
4. **合否閾値（事前宣言）**：**(i) DSR ≥ 0.95 かつ (ii) ネット Sharpe>0**、**かつ副条件 (iii) 廃止封筒の
   全−100% 端でもネット Sharpe>0**（恣意的閾値でない頑健性の定性条件）。(i)(ii)(iii) 全て満たせば PASS、
   いずれか欠ければ FAIL/脆弱。**FAIL は正当な結論**（docs/20 に正直に記録）。
5. **廃止封筒**：採用族の Sharpe を **last-price と 全−100% の両端**で併報（§4-(iii) 副条件の基礎）。
6. **レジストリ**：scope `phase4_gkx_judgment` に **判定を 1 件**だけ事前登録（preregister）＋結果記録
   （record_result・一度きり・改竄不能）。**K は +1（下した判定の本数）**。多重検定は N=6 で DSR 側が吸収
   （K+=6 は二重補正のため**しない**）。DSR は外部算出（N=6・V[パス]）して extra に保存。二重実行ガードで
   scope に既存試行があれば中止（K 水増し＝p-hacking を防ぐ）。

## 6. 現実性・頑健性（throwaway 診断・判定の試行数には算入しない）

handoff §7。判定の選択には使わない（事後の再選択禁止）。docs/20 に併記：
- (a) 業種変更銘柄を除く部分標本（sector-PIT 影響）／(b) 全上場診断での除外テールのアルファ量
  （mcap 床調整材料）／(c) レジーム別（利上げ局面 等）／(d) ホライズン（1M シグナルを 1M vs 3M 保有）／
  (e) mcap 床感応度（¥100億 vs ¥30–50億）／(f) 回転率・容量。

---

## 7. 確定事項（ユーザー確認反映・2026-06-21）

OOS 結果を見る前にここで凍結する。**行1・行2は前回確定（Combinatorial k=2／K+=1）に修正済み**。

1. **CPCV 分割**：**Combinatorial `n_splits=6・n_test_splits=2`・φ=5 パス/族**。V[SR]＝採用族の**パス間分散**
   （CPCV を使う動機）。`gap_before=1ヶ月`（ラベル）＋ `gap_after`＝embargo（§下 ★）。
2. **デフレート試行数 N／K**：**N=6（モデル族）で DSR デフレート**・sr_variance＝採用族の**φ=5 パス間分散**。
   レジストリは **判定 1 件のみ登録＝K は +1**（φ・N は K に算入しない＝二重補正回避）。
3. **マクロ系列**：**`{jp_10y, term_spread, vix, n225_iv, foreign_flow}` の 5 系列**（ユーザー確定）。
   credit spread は不在で除外。N225 IV は options_225 ATM IV の日次集約（状態変数・新規特徴量ではない）。
4. **合否閾値**：**(i) DSR ≥ 0.95 ∧ (ii) ネット（手数料後）Sharpe > 0 ∧ (iii) 副条件＝廃止封筒の
   全−100% 端でもネット Sharpe > 0**。全て満たせば PASS。(iii) は頑健性の定性条件（恣意的閾値ではない）。
5. **OOS 判定期間**：**2018-01 以降**（学習は全期間・判定は被覆安定後）。
6. **取引コスト**：片道 **15bps**・L/S 上位/下位**デシル**・月次・等加重。
7. **mcap 床**：本番 **¥100億**固定（docs/16 除外テール診断は robustness で併報）。

### ★ embargo（`gap_after`）＝ **12ヶ月で確定（A・ユーザー承認 2026-06-21）**

最長の特徴量後ろ向き窓は mom_36m/residual_mom=36ヶ月、seasonality=最大10年。train obs の特徴量窓が
テスト期間に届く境界リークを purge/embargo の後側ギャップ `gap_after` で遮断する。seasonality 完全被覆
（120ヶ月）は学習標本を消すため、**因果・PIT な特徴量は前向きラベルを直接漏らさない**ことを踏まえ
**(A) `gap_after=12ヶ月`** を採用：直近1年の系列相関＋中短期特徴量窓を遮断し、mom_36m/seasonality の
深い尾は「因果特徴は前向きラベルを漏らさない」として許容（学習標本の損失は限定的）。
`tests/test_phase4.py`（`test_purge_train_positions_*`）で「`gap_after` ぶんの後続 train 月が purge される」
ことを担保済み（36ヶ月窓の除去も検証）。`run_phase4_judgment.py` の `GAP_AFTER=12`。

> 流れ：最終 GO → 本書 FROZEN → `examples/run_phase4_judgment.py` を**一度だけ**実行 →
> `docs/20-phase4-results.md`（OOS R²・線形 vs ML 増分・採用族の deflated Sharpe＋廃止封筒・頑健性・
> 正直な合否）→ `registry_status.py` で K=+6 を確認。**以後、通すための変更はしない。**
