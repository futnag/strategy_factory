---
name: operator
description: 自律運用のディスパッチャ（1起動＝状態把握→優先規則で次の1アクションを決定→実行→ブリーフ）。ロック取得・loop_lint ゲート・PENDING 尊重・予算管理を内蔵。無人スケジュール起動（headless claude -p "/operator"）の唯一のエントリポイント。引数 plan で判断のみ（実行しない）。
---

# /operator — 自律ループのディスパッチャ

**役割**: 「次に何をすべきか」の判断を成文化した状態機械。1起動で**原則1アクション**
（束ねてよい組は明記）を実行し、日次ブリーフを書いて終わる。

## ガードレール（違反不可）

1. **ロック**: 開始時に `research_ops/operator.lock` を確認。存在し新鮮（<6h）なら
   **即終了**（並行実行禁止）。stale（≥6h）なら中身をブリーフに記録して上書き取得。
   取得時に `{"holder": "operator", "started_at": "<ISO>", "action": "<予定>"}` を書き、
   終了時に必ず削除（異常終了への備え＝次回の stale 検知が保険）。
2. **loop_lint ゲート**: ロック直後に必ず実行。
   - exit 1（HARD）→ PENDING に incident 追記・ブリーフ作成・**即停止**（何も実行しない）
   - exit 2（BUDGET）→ 研究サイクル系（red-team→cycle・replicate）は選択肢から除外
   - exit 0 → 全アクション可
3. **PENDING 尊重**: open の P-n に依存するアクションは選ばない（例: monitor の認定値が
   TO_CONFIRM でも monitor 自体は走らせてよいが、判定の格上げ提案はしない）。
4. **人間ゲートは越えない**: PASS 採用・Phase 2 操作・§A 変更・K 上限変更・ToS グレーの
   取得は、実行せず PENDING 追記＋通知のみ。
5. **1セッション予算**: アクション1件＋ブリーフで終了（データ更新＋QA は1件と数える）。
   セッション上限に当たったら: 状態を operator_log に書き、ロックを削除して正常停止
   （途中の子エージェントは次回 SendMessage 再開）。
6. **通知**: incident / PENDING 新規追加時は、可能な手段で人間に通知
   （PushNotification が使えれば送る・gh が使えれば ops リポの流儀で Issue 起票・
   どちらも不可ならブリーフ冒頭に ⚠ を立てる）。

## 優先規則（上から評価し、最初に該当した1つを実行）

| # | 条件（判定方法） | アクション | モデル指針 |
|---|---|---|---|
| 0 | loop_lint HARD | PENDING+通知して停止 | — |
| 1 | operator.lock が stale だった | 前回異常終了の調査をブリーフに記載（継続可） | — |
| 2 | データ鮮度: jquants daily の最新 < 直近営業日−2（loop_status で判定） | `/data-update`（QA 検収込み） | 安価で可 |
| 3 | 月が変わって最初の起動（monitor_log の最終 run 月 < 今月） | `/strategy-monitor` → `/ops-review`（束ねて1件） | 安価で可 |
| 4 | data-qa の最終実行が7日以上前（data_qa_log で判定） | `/data-qa` | 安価で可 |
| 5 | BUDGET でない ∧ BACKLOG ⬜≥1 | 先頭 ⬜ に `/red-team`（未レビュー時）→ approve/revise反映 なら `/research-cycle`（束ねて1件） | **設計系＝強いモデル推奨** |
| 6 | BACKLOG ⬜<2 | `/research-scout` | 設計系＝強いモデル推奨 |
| 7 | data_ideas 待ち行列に ToS クリアな取得タスクあり | `/data-acquire`（1件） | 中 |
| 8 | データ待ち IDEAS ≥3 ∧ data_ideas に新顔なし | `/data-scout` | 中 |
| 9 | 四半期初 ∧ retro 未実施 | `/loop-retro` | 中 |
| 10 | 該当なし | ブリーフのみ（「なにもすることがない」と正直に書く） | — |

## 手順

1. ロック取得（ガードレール1）→ `loop_lint` → `loop_status.py`＋research_ops の各 log の
   最終実行時刻を確認 → PENDING の open を読む。
2. 優先規則で1アクション決定。**引数に `plan` があればここで判断根拠を出力して終了**
   （実行しない・ロック解放）。
3. アクション実行（対象スキルの SKILL.md に完全準拠。operator は手順に介入しない）。
4. **日次ブリーフ** `research_ops/briefs/YYYY-MM-DD.md`:
   ⚠通知事項 → 実行アクションと結果1行 → PENDING open 一覧 → 明日の見込みアクション →
   予算状況（週次 K・データ鮮度）。既存same-dayブリーフがあれば追記。
5. `research_ops/operator_log.jsonl` に1行 append（decision・rationale・result・duration）。
6. ロック削除 → `git add research_ops/` → `ops(operator): <日付> — <アクション> <結果1行>`
   （アクション内のコミットは各スキルが行う＝operator は research_ops 分のみ）。

## 運用メモ（スケジュール設定＝人間が一度だけ）
- Windows タスクスケジューラ: 平日 22:30 に
  `claude -p "/operator" --dangerously-skip-permissions` 相当の headless 起動
  （権限は事前に settings 許可リスト整備を推奨）。cloud スケジュールはローカル data/ に
  届かないため不可。
- 週次 K 上限（loop_lint_baseline.json）が自律サイクルのスロットル＝実質「週3サイクル」。
  変更は人間のみ。
