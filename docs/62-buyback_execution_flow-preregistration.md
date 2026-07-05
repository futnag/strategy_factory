# 62. 自社株買いの実執行フロー（buyback_execution_flow）— 事前登録

**scope=`buyback_execution_flow`・K≤6（正式グリッドは Stage-0 生存時のみ）・2026-07-05 事前登録**
（起源: /research-scout IDEAS I-51 = Clarke 2022 FRL / BACKLOG buyback_execution_flow。
**red-team 2026-07-05 verdict=revise（非ブロッカー）を全反映**＝research_ops/redteam/2026-07-05-buyback_execution_flow.md。
サイクル `2026-07-05-buyback-execution-flow`）

---

## §0 既知の失敗型の構造的回避（red-team 反映）

| 失敗型 | 本設計での回避／自認 |
|---|---|
| **F7 発表軸 redux** | 既試 shareholder_return（**発表**ドリフト❌ DSR0.24・PBR改革後反転）と識別必須。本件は取締役会決議（発表）でなく**月次「自己株券買付状況報告書」＝実執行**がシグナル。統制で「執行フローの増分」を検定 |
| **H-16 同時性** | 買付は当月リターンと同時（買いが上昇の一因）→ **forward 必須**（提出月 t → **t+1** リターン。当月不使用） |
| **F2/H-4 現レジーム内・認定不能** | ローカル収録は 2025-06+ の **~13月次断面のみ**＝全期間 2024+ ＝OOS/サブ期間分離不可・minTRL 不足。**認定でなく機構実在判定**（§2.6） |
| **H-15/F6** | Clarke は米＝JP 方向は「執行 persistence（複数月の価格非感応な大口買いの継続）」で事前固定。買戻アノマリーは国際的に弱い（MPW）＝gross 薄い事前分布 |
| **H-18 統制** | プラセボ（同数のランダム・バスケット）＋サイズマッチで「執行フロー特異」を露出変化から分離 |
| H-3 コスト床 | ロングオンリー・借株不要・中大型＝月次XS 15bps で成立余地 |

## §1 実現可能性（記述スキャン・K=0・実施済み）
- 源: **EDINET「自己株券買付状況報告書（法24条の6第1項）」**（`data/edinet/list/` docDescription フィルタ）。
  ※BACKLOG の「TDnet タイトル」は誤り（ローカル TDnet は6週のみ）＝red-team ①で訂正。
- 実測: **6,618 filings・2025-06〜2026-06（13ヶ月）・distinct 1,267 社・secCode 非欠損 98%（5桁＝J-Quants 直結）**。
  月次 ~500-590 filings（プログラム継続中は毎月提出＝persistence）。
- **アンカー = submitDateTime**（`periodStart/End` は list メタで全 NULL＝本文/XBRL のみ）。提出月 t で「執行中」を確定。
- 価格: `data/processed/equities/wide/adj_close.parquet`（2016-06+・5桁コード列）＝月次リターンに resample。
- **カバレッジ限界**: 2024 の大量開示期は未収録（in-regime のみ）。powered 検定は D-8（EDINET 2016-2024 backfill）待ち。

## §2 事前登録グリッド（Stage-0 生存時のみ・5セル・K≤5）

| # | strategy_id | 定義 | 検定 |
|---|---|---|---|
| 1 | bef_exec_ls | 提出月 t の executor バスケット（EW）ロング − ユニバース EW ショート・t+1 | 主セル（執行フロー） |
| 2 | bef_exec_sizematch | 同・ショート脚をサイズ十分位マッチ | 統制（サイズ交絡） |
| 3 | bef_exec_costed | 1 に月次XS 15bps | H-3 コスト床 |
| 4 | bef_placebo | 同数ランダム・バスケット（各月）× N シードの null | H-18（フロー特異の識別） |
| 5 | bef_persist_w2 | executor が2ヶ月連続提出（強い persistence）× t+1 | 機構の順序（persistence の濃淡） |

- judge 配線（Stage-0 生存時）: `judge_grid(scope="buyback_execution_flow", 月次系列, costs_bps=15,
  execution_lag=1, registry=default_registry())`。

### §2.6 合格基準（事前固定・H-14/H-4 セマンティクス）
1. **認定でなく機構実在判定**（13月・in-regime＝minTRL 不足を事前受容）。標準 DSR≥0.95 は表示。
2. **Stage-0 必須ゲート（実行前・K=0）**: executor バスケットの t+1 forward 超過（対ユニバース）が
   **(i) 正 (ii) プラセボ分布の外 (iii) サイズマッチでも正**。いずれか未達で **KILL**（グリッド不要）。
3. 副次: 発表軸（shareholder_return❌）との符号・大きさの差（執行軸の増分）。

## §3 実装ステージ
1. **Stage-0（実行前 K=0 キルゲート・必須）**: 上記 §2.6.2 の forward 超過・プラセボ・サイズマッチ。
2. **Stage 1（Stage-0 生存時のみ）**: §2 の5セルを judge_grid で1回判定。
3. 診断（throwaway・K不変）: 月別 spread・persistence 濃淡・コスト感応。

## §4 出典
- Clarke (2022) "It's just a matter of time: Abnormal returns after firms stop repurchasing shares"
  Finance Research Letters 49, DOI 10.1016/j.frl.2022.103113（確認済・米・停止後に正の異常リターン＝執行フローが価格形成）。
- red-team 2026-07-05（verdict=revise・非ブロッカー）: データ源訂正（TDnet→EDINET）・in-regime 再枠付け・
  H-16 forward・F7 統制・Stage-0 K=0 義務化（research_ops/redteam/2026-07-05-buyback_execution_flow.md）。
- 既試 shareholder_return（発表ドリフト❌ DSR0.24・docs/03 §6.17）との識別が F7 の核。

## §5 結果（2026-07-05 判定）
**❌ FAIL — 最良 bef_exec_ls でも DSR=0.80 < 0.95。超過は執行フロー特異でなく size/liquidity 交絡。**

**Stage-0（K=0・生存）**: 13月（2025-06→2026-06・executor 6,510 filings/1,267社）。
生spread(exec−univ) **+0.38%/月**（IR1.39・hit69%）／プラセボ片側 **p=0.018**（null の外）／
turnover十分位中立 **+0.12%/月**（IR**0.40**・hit54%）＝3ゲート全通過 → grid へ。

**Stage-1 judge_grid（K=3・costs15bps・execution_lag=1・adv容量・値幅ロック）**:

| strategy | SR(ann) | PSR | **DSR** | minTRL(月) |
|---|--:|--:|--:|--:|
| bef_exec_ls（対ユニバース） | +1.54 | 0.94 | **0.80** | 261 |
| bef_persist_w2（2ヶ月連続） | +1.20 | 0.89 | **0.68** | 437 |
| bef_exec_sizematch（turnover中立） | −0.07 | 0.47 | **0.21** | ∞ |

**αの帰属（決定的）**: raw の bef_exec_ls は DSR0.80（本ループ全試行で trend_structure 0.92 に次ぐ接近）
だが、**turnover十分位中立にすると SR −0.07・DSR 0.21**＝executor バスケットの超過は**執行フロー特異でなく
size/liquidity ティルト**（自社株買い執行企業＝大型・高流動で、当窓で大型が優位）。Stage-0 の turnover中立
+0.12%（IR0.40）が既に foreshadow し、judge の 15bps コストがそれを負に落とした。**gross 機構は size 交絡で
説明し尽くされ、執行フロー機構は非確認**。in-regime 13月＝minTRL 261月に遠く及ばず認定は構造的に不能。

**失敗型**: F7亜型（size/liquidity 交絡）。**新規教訓 H-22**（イベント・バスケット生LSの size/liquidity 交絡＝
size中立セルで帰属を確定せよ）を HEURISTICS に追加。
**フォローアップ**: D-8（EDINET 2016-2024 backfill）で powered 化しても、size中立αが無い限り再訪不可
（打ち止め寄り）。発表軸 shareholder_return❌（DSR0.24）に続き、執行軸も否定＝自社株買い3軸目も棄却。
