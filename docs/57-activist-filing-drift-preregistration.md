# 57. 大量保有報告イベントの filer 異質性ドリフト（activist_filing_drift）— 事前登録

**scope=`activist_filing_drift`・K=4（グリッド4セル・extra_trials=0）・2026-07-03 事前登録**
（起源: /research-scout 2026-07-03 調査＝research_loop/IDEAS I-33・BACKLOG 昇格。
サイクル `2026-07-03-activist-filing-drift`）

---

## §0 既知の失敗型の構造的回避（LESSONS §1 との突合）

| 失敗型 | 本設計での回避 |
|---|---|
| F1 直近regime共変 | ⚠ 本丸リスク（アクティビストは低PBR選好・2023+ ガバナンスラリー）→ サブ期間
  符号 2/3 を副次基準＋value 因子との相関を診断必須表示。標本が 2022+ のみ＝「直近だけ効く」の
  検定力は限定的である旨を §2.7 に明記 |
| F3 符号逆 | 方向は日本主標本（Gillan et al. 2023: ファンド系で正・PBFJ）＋JP 先行 CAR +1.5-2% で
  ロング側に事前固定。逆なら FAIL |
| F4/H-16 | 蓄積セル（変更報告の持分増）は「Δ系」＝同窓の momentum/リバーサルとの相関を診断必須
  （H-16 の初適用）。イベントアンカーゆえ構築窓は短い（前回報告比）＝直交性 prior は中立 |
| F5 de-risk | イベント名ロング×流動ヘッジショートのドルニュートラル＝該当なし |
| F7 redux | 既試「アクティビスト静的レジストリ」は銘柄スクリーン＝本件は**報告イベントの
  ドリフト**で軸が別。tob_arb とも別（TOB は個別の裁定・本件は 5% 報告一般） |
| F8/H-12 | 新規報告 ~1,000件/年・保有40営業日＝同時保有 数十〜百数十銘柄で breadth 確保。
  対象の 70%超が Prime（JRI 2026）＝micro-cap 集中でない |
| F9 データ制約 | §1 で確認済み。sec_code 欠損（90%）は edinet/list の edinetCode→secCode
  対応表（2,610日分 union）で解決（実装内の配管・K=0） |

## §1 実現可能性（記述スキャン・K=0）
- `data/edinet/large_holdings.parquet`（新規 4,503件）＋`large_holdings_changes.parquet`
  （変更 3,226件）: **2022-01-04〜2026-06-12**（約4.4年）・年 1,100〜1,900件で安定。
- PIT アンカー: `submit_dt`（時刻付き提出タイムスタンプ）。
- filer 分類は**既存の名簿で確定済み**: `in_registry`（アクティビスト名簿・2,626件）・
  `style`（engagement_hard 2,278 / soft 348）・`is_important_proposal`（重要提案行為・2,687件）
  ＝分類ルールを本サイクルで新設しない（恣意性排除）。
- `sec_code` 直接収録は 775/7,775 のみ → `issuer_edinet` から edinet/list メタデータの
  edinetCode→secCode 対応（上場発行体は自社提出書類で secCode 付き）で復元する。

## §2 事前登録（固定・実行前コミット）

### §2.1 仮説（方向は日本主標本で事前固定・全てロング側の正ドリフト）
- **H1**: アクティビスト名簿 filer（`in_registry`）による**新規**大量保有報告後、対象銘柄は
  1〜2ヶ月アウトパフォーム（エンゲージメント期待の段階的織り込み）。
- **H2**: `is_important_proposal`（重要提案行為等）付き報告後のドリフトは特に正。
- **H3**: 名簿 filer の**積増し**（変更報告で持分比率 +1pt 以上）後も正のドリフト（継続確信の表明）。
- 統制: 全 filer の新規報告セル（filer 異質性＝H1 が統制を上回るかを §5 に記録）。

### §2.2 経済的根拠
大量保有報告は「誰が・何の目的で」ブロックを取得したかの公開情報だが、エンゲージメントの
帰結（還元・ガバナンス変化・再評価）は不確実で段階的に実現するため、開示時点で完全には
織り込まれない（Gillan et al. 2023 の異質性・JP 先行研究の CAR +1.5〜2%＝米国より小さい初期反応
＝ゆっくり織り込まれる余地）。

### §2.3 データと PIT 規律（チェックリスト確認済み）
- イベント日: `submit_dt` の日付（t）。ポジション開始は **t+1 営業日**（提出は場中〜夕方のため）。
- 日次イベント・ポートフォリオ: 窓内（開始から40営業日）のイベント銘柄を等加重ロング、
  流動ヘッジ（trailing ADV 上位300・EW）ショートのドルニュートラル（margin_alert/docs/53 と
  同機構のロング側）。イベント0件日は現金。
- PIT eligible（trailing252日≥120日約定）・`limit_lock_flags`・adv=trailing 売買代金・
  participation=0.1・貸株115bps（ヘッジ側・流動大型）。コスト片道30bps（日次イベント床）。
- OOS(2024-01+)は設計不使用（診断のみ・標本 2022+ ゆえ IS も短い旨併記）。

### §2.4 シグナル構築＝固定グリッド（4セル・K=4・保有 H=40営業日）

| # | strategy_id | イベント母集団 | 検定 |
|---|---|---|---|
| 1 | `act_new_all_h40` | 新規報告・全 filer（sec_code 復元可能分） | 統制 |
| 2 | `act_new_reg_h40` | 新規報告・`in_registry` filer | **H1 主セル** |
| 3 | `act_new_prop_h40` | 新規報告・`is_important_proposal`=True | H2 |
| 4 | `act_accum_reg_h40` | 変更報告・`in_registry` かつ 比率 +1pt 以上 | H3 |

### §2.5 judge 配線
`judge_grid(scope="activist_filing_drift", costs_bps=30, execution_lag=1, adv=trailing売買代金,
participation=0.1, no_buy/no_sell=limit_lock_flags, short_borrow_bps=115,
registry=default_registry())`（日次ビュー＝adj close・lag=1 で T+1 執行）。

### §2.6 合格基準（事前固定）
1. **DSR ≥ 0.95**（唯一の主基準）
2. 副次（PASS の追加条件）: (i) 最良セル net SR>0 (ii) サブ期間（judge 3分割）符号 2/3 以上正
   (iii) 容量 ≥ ¥5,000万
3. 仮説別記録（PASS/FAIL と独立に §5 へ）: H1 vs 統制（reg − all の SR 差の符号）・H2・H3 の符号。
4. 未達は機械的に FAIL。事後救済なし（filer 分類・窓 H・閾値の変更再実行禁止）。

### §2.7 既知の限界（正直に）
- **標本 4.4年（2022-01〜）**＝n≈1,090日。認定には強い SR が必要（H-4 域）。また 2023+ の
  ガバナンスラリーが標本の大半＝F1（value 共変）を標本内で切り分ける検定力は限定的。
  value 因子リターンとの相関・低PBR 銘柄比率を診断で必須表示。
- EDINET の 5% 報告保持期間により 2022 以前は取得不能（拡張不可の構造的制約）。
- 積増しセル（H3）は Δ系＝H-16 診断（同窓価格相関）を必須表示。
- 提出時刻が 15:00 以降の報告も t+1 執行＝一部は t+1 寄りに情報が既に反映され得る（保守側）。

## §3 実装ステージ（同一サイクル内で許される段階＝これのみ）
1. **Stage 1（本判定）**: §2.4 の4セルを judge_grid で1回判定。
2. 診断（throwaway・K不変）: gross/net・IS/OOS(2024-01)・年次SR・コスト/貸株感応・
   イベントスタディ形状（t+1..t+60 平均超過累積）・ρ(戦略リターン, value因子)・
   対象銘柄の 低PBR比率/ADV 分布・H1−統制の差分系列・H3 の同窓 momentum 相関（H-16）。

## §4 出典
- Gillan, Nguyen & Nishikawa (2023) "Heterogeneity in shareholder activism: Evidence from Japan"
  PBFJ 77, 101891（abstract 確認済・日本主標本）
- Yoshida (2026) "The Rise of Shareholder Activism in Japan" JRI Research Journal（PDF 確認済:
  対象295社・70%超 Prime・2025年56キャンペーン）
- 既試との区別: アクティビスト静的レジストリ（スクリーン）・tob_arb（裁定）とはイベント軸で別
- 調査: research_loop/surveys/2026-07-03-crypto-event-jpnative.md（I-33）

## §5 結果（実行後に記入）
（未実行）
