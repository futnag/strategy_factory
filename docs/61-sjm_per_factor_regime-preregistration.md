# 61. per-factor regime 切替（sjm_per_factor_regime）— 事前登録

**scope=`sjm_per_factor_regime`・K≤6（正式グリッドは Stage-0 生存時のみ）・2026-07-05 事前登録**
（起源: /research-scout IDEAS I-12 = Shu & Mulvey 2024 arXiv:2410.14841 / BACKLOG sjm_per_factor_regime。
**red-team 2026-07-05 verdict=revise（非ブロッカー）を全反映**＝research_ops/redteam/2026-07-05-sjm_per_factor_regime.md。
サイクル `2026-07-05-sjm-per-factor-regime`）

---

## §0 既知の失敗型の構造的回避（red-team 反映・LESSONS §1 突合）

| 失敗型 | 本設計での回避／自認 |
|---|---|
| **H-5/F5 de-risk** | regime exit＝露出調整＝Sharpe 改善なしの疑いが最有力（~55%）。**常時オン（買い持ち等ウエイト）超過を主眼の副次基準**にし、統制セルとして常時オンを同梱 |
| **H-18 統制** | **placebo=regime ラベルをシード固定でシャッフル**（露出割合は保存・timing のみ破壊）を必須セル化＝「切替の情報価値」と「露出変化の副産物」を分離 |
| **H-1 momentum 代理** | regime≈因子自身の TS モメンタム。**素朴 mom オーバーレイ（trailing-12m>0）統制**を同梱し、SJM が単純トレンドを超えるかを識別 |
| **H-11 複雑度vsデータ** | red-team の核。→ **FF Japan 3因子（430ヶ月）を採用**（repo ローカル 2016+ の ~120月でなく）＝regime 推定に十分な標本を与え「データ不足で死んだ」交絡を排除。遷移数を Stage-0 で記述カウント |
| **H-14 改良DSR張り付き** | これは固定合成の改良研究＝standalone DSR は張り付く。**§2.6 で認定でなく帰無再評価と明記・PASS 相当でも proposals 行き** |
| **H-15 US符号反転** | 出典は US 6因子。JP 転移は方向・大きさとも額面不可＝機構テストとして扱う |
| PIT リーク | regime 推定は**オンライン拡張窓（regime_t は data≤t のみ）・配分 T+1（execution_lag=1）**。全標本 Viterbi 平滑（未来参照）を禁止。ジャンプペナルティ・状態数・因子集合・特徴を全事前固定 |
| p-hack | ペナルティ/状態数/ルックバックのグリッド探索を禁止＝**単一値事前固定**（下記） |

## §1 実現可能性＋PRE-FIXED 設計（記述スキャン・K=0）

- **factor set（事前固定・後知恵選抜なし）**: `data/external_factors/ff_japan_3factors.parquet` の
  **{Mkt-RF, SMB, HML}** 月次超過リターン。**430ヶ月 1990-07..2026-04**（外部・Kenneth French Japan）。
- **model（事前固定）**: 2状態ジャンプモデル（座標降下）。特徴 = z[月次リターン], z[roll-6m 実現ボラ]。
  **ジャンプペナルティ LAMBDA=8（固定・非チューニング）**。high-mean 状態を good と online 命名。
- **inference（事前固定）**: オンライン拡張窓。`MIN_TRAIN=120`（10年）後に regime を推定＝評価期間
  **2000-07..2026-04（310ヶ月）**。regime_t は data≤t のみ・配分は shift(1)＝T+1 執行。
- **allocation（事前固定）**: 各因子 good regime なら 1・else 0 → 稼働因子で等ウエイト再正規化（全 off は現金）。
- **統制（事前固定・H-18/H-1/H-5）**: 固定等ウエイト（常時オン）／mom オーバーレイ（trailing-12m>0）／
  placebo（regime ラベルをシード固定でシャッフル・50シードで分布評価）。

## §2 事前登録グリッド（Stage-0 生存時のみ・5セル・K≤6）

| # | strategy_id | 定義 | 検定 |
|---|---|---|---|
| 1 | sjm_switch | SJM online regime 切替合成（主） | 改良の主セル |
| 2 | sjm_fixed_ew | 固定等ウエイト（常時オン）＝改良の分母 | H-5 買い持ち基準 |
| 3 | sjm_mom_overlay | trailing-12m>0 オーバーレイ | H-1 統制（素朴 mom 超えるか） |
| 4 | sjm_placebo | regime ラベルシャッフル | H-18 統制（timing 情報の有無） |
| 5 | sjm_switch_costed | 1 に切替コスト（月次XS 15bps・回転実測） | H-3 コスト床 |

- judge 配線（Stage-0 生存時）: `judge_grid(scope="sjm_per_factor_regime", 月次系列, costs_bps=15,
  execution_lag=1, registry=default_registry())`。因子系列レベル戦略（stock 非依存）。

### §2.6 合格基準（事前固定・H-14 セマンティクス）
1. **これは認定でなく「固定等ウエイト合成に対する改良の帰無再評価」**。標準 DSR≥0.95 は主基準だが、
   改良研究ゆえ standalone DSR は分母に張り付く見込み（H-14）＝**PASS 相当でも認定でなく proposals
   （人間ゲート）行き**。判定の第一目的は「切替が固定を上回るか（改良幅とその実在性）」。
2. **Stage-0 必須ゲート（実行前・K=0・下記 §3）**: これを通らなければ正式グリッドは回さない。
3. 副次（改良の実在条件）: (i) switched SR > 固定等ウエイト SR（F5 否定）
   (ii) switched が **placebo 分布（シャッフル）の外**＝timing 情報あり（H-18）
   (iii) switched が mom オーバーレイと同等以上（H-1＝単純トレンド以上の価値）。

### §2.7 既知の限界（正直に）
- FF 3因子は repo の本番スリーブ（value+PEAD/TSMOM）そのものではない＝**機構テスト**であり、
  仮に生存しても本番合成への適用は別途 proposals。
- 月次×3系列＝breadth 小・minTRL は長い（H-4）＝認定は構造的に困難（改良幅の実在判定が主目的）。

## §3 実装ステージ（同一サイクル内で許される段階＝これのみ）
1. **Stage-0（実行前 K=0 キルゲート・必須）**: red-team 義務化の2本＋統制比較。
   (a) H-11: 各因子 in-sample regime 遷移数（~1 なら過適合棄却）
   (b) F5/H-18/H-1: switched SR vs 固定・vs placebo 分布・vs mom。
   **F5（固定未満）or H-18（placebo 同値）or H-11（遷移~1）or H-14（rho>0.9∧エッジ無）のいずれかで KILL**
   ＝グリッド不要の documented negative。
2. **Stage 1（Stage-0 生存時のみ）**: §2.4 の5セルを judge_grid で1回判定。
3. 診断（throwaway・K不変）: 系列別 Sharpe・遷移プロファイル・コスト/回転感応。

## §4 出典
- Shu & Mulvey (2024) "Dynamic Factor Allocation Leveraging Regime-Switching Signals" arXiv:2410.14841
  （US 6 スタイル因子・2状態 SJM・IR 0.05→0.4-0.5・確認済 IDEAS I-12）。
- red-team 2026-07-05（verdict=revise・非ブロッカー）: H-14 判定意味論・H-18 統制4セル・PIT/オンライン・
  パラメータ事前固定・Stage-0 キル2本義務化（research_ops/redteam/2026-07-05-sjm_per_factor_regime.md）。
- 既存 prior: H-5（de-risk）・H-1（mom 代理）・H-11（複雑度）・H-15（US 符号反転）・LESSONS §3（2023+ レジーム敵対）。

## §5 結果
**未実行**（事前登録コミット時点）。
