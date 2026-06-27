# 46 — 条件付き no-arbitrage SDF 事前登録（候補②：regime-conditioning の"正解" ＋ GKX の切り口変更）

**作成日**: 2026-06-27
**位置づけ**: docs/45（テキストα cheap版＝FAIL）に続く候補②。最新研究サーベイ（docs/03 §6.20 関連・deep-research
2026-06-27）の**最重要発見**＝「私のレジーム/de-risk 失敗の"正解"」を実装・検定する。

---

## 0. なぜこれか（失敗型の正面修正）

9連敗のうち **①option-regime / ③flow / ⑥FX-rotation は "de-risk止まり・naive regime"**、**GKX(キッチンシンクML)は帰無**。
サーベイがその**原因と処方箋**を特定した（Chen-Pelger-Zhu, "Deep Learning in Asset Pricing", SSRN 3350138・◯敵対的検証済）:

- ◯ **生マクロ/最新値を素朴に条件付けると OOS Sharpe が完全崩壊**。最新の増分だけでは景気循環のような**動的状態**を表せない。
  正解＝**recurrent net で潜在経済状態を抽出**し、それを**no-arbitrage SDF の目的関数自体に入れる**（重みでなく）→ 無条件版比 **+約20% Sharpe**。
  ＝**私の実現/インプライド・ボラ・ゲート全敗の理由そのもの**（最新ボラ1点で条件付けた＝動的状態でない）。
- ◯ **GKX 型"予測"の高Sharpeは小型株アーティファクト**（value-weight で約−50%）。だが**no-arbitrage SDF は大型でも生存**
  （1,500大型で OOS Sharpe **1.4**／550大型で **0.9**、線形/予測モデルは崩壊）。
  ＝私の **GKX 帰無は「予測目的関数＋小型依存」が原因**の可能性＝**切り口＝SDF目的・value-weight・大型**で再挑戦すべき。

＝これは新factor探しでなく、**「目的関数」と「条件付けの作法」を変える**＝既試の失敗を構造的に避ける。

---

## 1. 実現可能性（既に使い込んだデータ＝確立済み）

- **特性 Z**（PIT・セクター中立z）: value(B/M,E/P,CF/P,S/P)・quality(roe/roa/margin/accruals)・size・momentum・low-vol・
  配当利回り ＋ 信用/空売り(short_to_long, short_interest, margin_imbalance) ＋ フロー。`value_quality_size_factors` 他で算出済。
- **マクロ M**: `load_macro`(jp_10y/jp_policy/us_10y/us_ff/cpi/vix)＋`load_external_prices`(usdjpy/eurjpy/audjpy/商品)＋n225_iv。~15-20系列。
- **ユニバース**: 大型（時価総額上位・value-weight）。PITユニバース＋mcap で構築可。月次・フォワード月次リターン。
- **計算**: 最小版は **numpy/sklearn のみ**（後述）。深層SDF（torch）は後段 upgrade。

---

## 2. 事前登録：仮説と方法（**走らせる前に固定**）

### 2.1 中核アイデア（用語）
- **特性管理ポートフォリオ** F_t = Z_{t-1}' R_t（各特性 × 横断リターン＝1ファクター時系列。先読み無し）。
- **no-arbitrage SDF**: M_t = 1 − b' F_t。b は**接線ポートフォリオ係数**＝KNS（Kozak-Nagel-Santosh 2020）の
  **二乗誤差＋L2縮小**で推定（pricing error 最小化＝no-arbitrage 目的。`numpy`/`sklearn ridge`）。
- **潜在経済状態** S_{t-1}: マクロ M の**動的要約**（最新値でなく trailing 3/6/12m 変化＋12m ロール z → PCA で数本）。
  ＝サーベイの「hidden state」を numpy で近似。**負のコントロール**は「M の最新観測のみ」。

### 2.2 仮説（falsifiable）
- **H1（SDF が予測 MLを上回る）**: 大型・value-weight 上で、KNS no-arbitrage SDF（接線）の OOS Sharpe が
  (a) GKX 型予測 L/S、(b) FF/単純合成、(c) 私の既試 multifactor を上回り、**デフレートDSR を有意に上げる**。
- **H2（条件付けの作法）**: **潜在状態 S で条件付けた SDF**（b を S の関数に：b_t = b0 + B·S_{t-1}）は無条件SDFの OOS を上回る。
  一方 **最新マクロ1点で条件付けると OOS が崩壊**（負のコントロール＝サーベイの再現＝"なぜ私のゲートが失敗したか"の実証）。
- **H3（容量＝小型アーティファクトでない）**: SDF の OOS Sharpe は **value-weight/大型でも維持**（GKX予測は equal-weight/小型で半減）。

### 2.3 評価（既存規律）
walk-forward OOS（拡大窓・年次再fit）、purged/embargo（既存 CPCV）、**デフレートDSR**（縮小パラメータλ・状態本数K等も試行計上）、
PSR/minTRL/サブ期間。大型ユニバースは value-weight。コスト/容量/執行ラグは既存エンジン。`examples/research_conditional_sdf.py`。

### 2.4 合格基準（事前固定）
1. **OOS デフレートDSR ≥ 0.95**（縮小・状態本数の格子を K に算入）。
2. **H1**: SDF が GKX予測・単純合成を OOS Sharpe で上回る（同一ユニバース・同一コスト）。
3. **H2 負のコントロール**: 最新マクロ条件付けが崩壊（OOS Sharpe が無条件以下）＝潜在状態の作法が効く証拠。
4. **H3**: value-weight/大型で OOS が維持（equal-weight/小型依存でない）。
5. **サブ期間**で SDF の符号一貫（直近のみ＝不採用）。

### 2.5 失敗型ガード
| 過去の失敗型 | 本設計での回避 |
|--|--|
| de-risk（α無し・①③⑥） | 横断SDFの接線で**α**を測る（指数タイミングでない）。状態は目的関数に入れる（重みのみでない） |
| naive regime（最新ボラ1点） | **潜在動的状態**で条件付け＋**最新マクロ＝負のコントロール**で作法の差を実証 |
| GKX帰無（予測＋小型） | **SDF目的＋value-weight大型**で切り口変更。H3で小型アーティファクトを排除 |
| 多重検定 | 縮小λ・状態本数K・特性集合を全て K 算入し**デフレートDSR**。GKXと同一の honest 勘定 |
| 直近のみ | サブ期間符号一貫を合格条件化、保留OOS 報告 |

---

## 3. 実装ステージ
1. **管理ポートフォリオ基盤** `invest_system/portfolio/sdf.py`：F_t 構築・KNS 接線（ridge縮小）・潜在状態 PCA・条件付き b。純関数＋テスト。
2. **Stage 1（最小・numpy/sklearn）** `examples/research_conditional_sdf.py`：無条件SDF / 潜在状態条件付き / 最新マクロ(負コントロール) / GKX予測 / 単純合成 を
   walk-forward OOS・大型 value-weight で対比 → judge（デフレートDSR）。H1/H2/H3 を一括検定。
3. **Stage 2（深層）**: torch で GRU 潜在状態＋FFN＋（任意で）GAN 的モーメント選択。依存追加の意思決定はここで。

---

## 4. 出典
- Chen, Pelger, Zhu — Deep Learning in Asset Pricing (Management Science 2024): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3350138 ◯
  （naive-macro 崩壊／潜在状態を目的関数へ／SDF が大型で生存・予測モデルは崩壊）
- Kozak, Nagel, Santosh — Shrinking the Cross-Section (JFE 2020) △（KNS 縮小SDFの方法論的土台・確立文献）
- AQR "Factor Timing is Hard" △ / Robeco factor timing △（regime/timing への実務の慎重論＝過度期待の戒め）
