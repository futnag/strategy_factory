# 63. PBR改革開示イベント条件付き value（governance_event_value）— 事前登録（Stage-0 cheap-kill）

**scope=`governance_event_value`・本サイクルは K=0（Stage-0 のみ・judge_grid を回さない）・2026-07-05 事前登録**
（起源: IDEAS I-9 = D'Ercole, Wagner & Yamada 2026 CEPR DP19971 / JCF 99。BACKLOG governance_event_value。
**red-team 2026-07-05 verdict=revise を全反映**＝research_ops/redteam/2026-07-05-governance_event_value.md。
サイクル `2026-07-05-governance-event-value`）

---

## §0 本サイクルの枠組み（red-team §A 反映）— 「窓の分離」

- 論文（I-9）の機構は **2024-01 salience shock**（対応一覧公表で低PBR×高ROE が一度に repricing）。
- 手元データ `data/tse_capital_disclosure/`（/data-acquire D-2）は **2025-05→2026-05 の13月＝shock は窓外**。
- ⇒ **論文 faithful な検定（2024窓）は D-7（backfill）取得後の別サイクル**に分離。
  **本サイクルは in-window tail（2025-06+ の新規開示）が生きているかの K=0 cheap-kill のみ**。
  予想は H-17/F6 で null＝その場合 **F6/H-17 negative を記録し scope を ⏸（解除条件＝D-7 取得）へ**。

### 失敗型の構造的回避（red-team 反映）

| 失敗型 | 本設計での回避／自認 |
|---|---|
| **F1 value/PBR改革共変** | 検定対象は生 value でなく **value 残差化後の増分**（月次 XS 回帰 forward~[BM,logturnover] の残差）|
| **F6 減衰＋H-17 即時織込** | 本 tail（2025+）は salience 抜け後＝**drift ~0 が帰無**。それを最安で確認するのが本サイクルの目的 |
| **H-22 size/liquidity 交絡** | 開示 firm は size に偏る（Std 新規=小型）→ **turnover 中立を残差に同梱**（buyback の教訓）|
| **H-16 同時性** | 開示月と同月リターンは同時（repricing が上昇原因）→ **forward t+1**（当月不使用）|
| **H-4 minTRL** | 13月 in-regime＝認定不能を事前受容。**K=0＝そもそも DSR を主張しない**（記述 cheap-kill）|
| H-3/F8亜型 | Std 小型・少数イベント＝容量/コストは Stage-0 生存時のみ judge で評価（本サイクルは gross 帰属のみ）|

## §1 実現可能性（記述スキャン・K=0・実施済み）
- イベント: パネルの**窓内 transition**（前月 非開示→当月 開示済＝初出月 > 2025-05）。左側打切り（2025-05
  時点 開示済・Prime ほぼ全部）は**除外**（PIT 偽装回避・red-team §D）。実測 ~290件（2025-06=121・07=49 に偏在・主に Standard）。
- 価格/サイズ: `adj_close`（月次）・`turnover`。value: `fundamentals_panel(me,["Eq","ShOutFY"])` → BM=Eq/(ShOut×close)。
- forward: 初出月 M（月末に status 既知）→ **M+1 リターン**。

## §2 Stage-0 設計（K=0・judge_grid 不使用）
各初出月 M のイベント・バスケットについて t+1 forward で3系列を算出:
1. **生 spread**（event − ユニバース）
2. **turnover 十分位中立**（H-22）
3. **value+size 残差**（月次 XS 回帰 `fwd_ret ~ rank(BM) + rank(log turnover)` の残差の event-basket 平均＝本命）
プラセボ（H-18）: 各月・同数のランダム・バスケット × N の null（残差平均の分布）。

### §2.6 合格基準（事前固定）
- **ゲート（KILL 条件）**: **value+size 残差の event-basket 平均**が (i) 正 かつ (ii) プラセボ null の外（片側 p<0.05）
  でなければ **KILL**（K=0）。→ KILL 時 verdict=FAIL（tail・条件付き）・scope ⏸（解除=D-7）。
- 生存時のみ（想定外）: 別サイクルで judge_grid（value 残差ロング×ユニバース・K≤6）＋ D-7 優先取得。
- **本サイクルで DSR は主張しない**（K=0・minTRL∞）。判定は上記ゲートのみ。

## §3 実装ステージ
1. **Stage-0（K=0・唯一の判定）**: §2.6 のゲート。**judge_grid は回さない**（後出し禁止＝生存時は新 prereg で別サイクル）。
2. 診断（throwaway・K不変）: 月別 event 数・生/turnover中立/残差 spread の月次系列・2025前半 vs 2026 の減衰比較。

## §4 出典
- D'Ercole, Wagner & Yamada (2026) CEPR DP19971 / J. Corporate Finance 99, 103009（RePEc 確認済）＝[IDEAS I-9]。
  機構=2024 salience shock の低PBR×高ROE repricing（**イベント時点の水準シフト**であって tradeable 後日 drift でない）。
- red-team 2026-07-05（verdict=revise）: 窓不一致（H-8/H-9）・H-17 drift 否定・H-22 size 交絡・value 残差化＋placebo・
  Stage-0 K=0 cheap-kill 化（research_ops/redteam/2026-07-05-governance_event_value.md）。

## §5 結果（2026-07-05 判定）
**KILL（K=0・tail 条件付き FAIL）→ scope ⏸（解除条件＝D-7 取得）。** ❌ではない（論文 faithful な 2024 窓は未検定）。

**イベント breadth（それ自体が所見）**: **cleanly-datable transition は 135件のみ**（前月 present-and-undisclosed→
当月 開示済＝PIT 安全に日付可能なもの。prev=NaN の 153件＝リスト新規入りは日付不能で除外）。**77件が
2025-06(61)/07(16) に偏在し以降 <10/月に崩落**（Standard 比率 66%）＝**salience 減衰の直接証拠**（H-9/F6/H-17）。

**ゲート（value+size 残差・forward t+1・turnover 中立込み）**:
- 主ゲート（月次・MIN_EV=10＝2月クリア）: 残差 **+0.98%/月**・placebo 片側 **p=0.223** → **ゲート(ii)未達＝KILL**。
- 診断（pooled・6月/103件・§3.2 throwaway）: 月加重残差 +2.12%（p=0.122）・全イベント等加重 **+0.75%**。
- **全 spec で点推定は微正だが placebo と区別不能**（p=0.12〜0.22）。tail が薄すぎて null を棄却できない。

**解釈**: 微正の点推定はノイズか faint な残光かを **13月 tail の breadth では判別不能**。これは (a) red-team の
F6/H-17 予測（2024 shock は減衰・遅れた開示に tradeable drift 無）と (b) データ窓問題（本命は 2024 窓＝手元に無）
の両方と整合。**tail cheap-kill は KILL**（judge_grid は回さず K=0）。**本命（低PBR×高ROE の 2024 salience
repricing）は D-7（2024-01〜2025-04 backfill）取得後に別 scope/prereg で検定**＝EV 最大の順序（red-team §I）。

**自己改善**: **H-23 新設**（salience/評判ショック機構は**イベント時点の水準シフト**であって tradeable な後日
drift でない・かつ post-shock の event breadth は崩落する＝**ショック窓そのものを取れ・余波の tail で代替するな**）。
本命の判別には D-7 が gating＝data_ideas 待ち行列で最優先化。
