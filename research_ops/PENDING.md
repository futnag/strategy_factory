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

### P-2 [investigate] Phase 2 成果物間の乖離（2026-07-04・/ops-review）
- equity_daily.csv の combo 累計 **−3.55%** vs status.json/months.csv **+0.61%**＝符号レベルの
  乖離。定義差（列のセマンティクス）か実ドリフトかの切り分けが必要。
  最短: phase2_reconcile.py の読み取り専用調査（Claude に依頼可）。
- 判断: ＜未記入＞

### P-3 [approve] 週次 K 上限の設定確認（2026-07-04・loop_lint）
- 既定 12試行/週（≈3サイクル）。今週は人力ビルドで32消費済み＝自律サイクルは
  次週まで待機が既定挙動。上限を変える場合は research_ops/loop_lint_baseline.json の
  k_weekly_limit を編集。
- 判断: ＜未記入＞

## Resolved

（なし）
