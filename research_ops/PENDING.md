# PENDING — 人間の承認/判断 待ち行列

**運用ルール**: `P-n` 番号制。スキル/オペレータが append し、人間が判断を追記して
`resolved` にする（削除しない）。**open の項目に依存するアクションは実行しない**。
種別: `approve`（PASS採用・platform変更・ToSグレー取得）/ `investigate`（要調査）/
`config`（凍結値の確認）/ `incident`（HARD違反・障害）。

---

## Open

### P-1 [config] monitor_config.json の認定値の凍結確認（2026-07-04・/strategy-monitor）
- 監視の帰無仮説（claimed_sr_ann 等）が seed 状態（TO_CONFIRM）。docs/03 §6.15-6.16 と
  突合のうえ status を CONFIRMED に更新してください（以後の変更は禁止＝攻撃面）。
- 判断: ＜未記入＞

### P-2b [approve] phase2_reconcile.py の1行修正の適用（2026-07-04・調査完了に伴う後継）
- **調査完了**（research_ops/ops_review_reports/2026-07-04-P2-investigation.md）:
  月次会計側のバグ＝`ext_cl.asof(val_date)` の全列非NaN巻き戻りでヘッジ/TS が
  **6/8 の古値評価**→ status は +1.11% だが真値 ≈ **−4.1%**（キルスイッチの目隠し状態。
  現時点は alert 閾値未達＝即時アクション不要）。
- 承認事項: ①`ext_cl.ffill().asof(val_date)` への修正（ローカル＋ops リポ両方）＋
  鮮度ガード追加 ②investers/ の更新停止12系列の扱い。Phase 2 コードは接触禁止圏のため
  **あなたの明示承認があれば私が修正＋検証まで実施**します。
- 判断: ＜未記入＞

### P-3 [approve] 週次 K 上限の設定確認（2026-07-04・loop_lint）
- 既定 12試行/週（≈3サイクル）。今週は人力ビルドで32消費済み＝自律サイクルは
  次週まで待機が既定挙動。上限を変える場合は research_ops/loop_lint_baseline.json の
  k_weekly_limit を編集。
- 判断: ＜未記入＞

## Resolved

（なし）
