# PENDING — 人間の承認/判断 待ち行列

**運用ルール**: `P-n` 番号制。スキル/オペレータが append し、人間が判断を追記して
`resolved` にする（削除しない）。**open の項目に依存するアクションは実行しない**。
種別: `approve`（PASS採用・platform変更・ToSグレー取得）/ `investigate`（要調査）/
`config`（凍結値の確認）/ `incident`（HARD違反・障害）。

---

## Open


### P-4 [approve] ops リポへの同修正の適用（2026-07-04・P-2b の残件）
- ローカルの phase2_reconcile.py は修正済み（コミット `eceb9fd`・P-2b 承認による）。
  **夜間 GitHub Actions（ops リポ）が同スクリプトを使っている場合は同じパッチの適用が必要**
  （それまでダッシュボードは古値ベースの表示が続く）。パッチ内容は
  ops_review_reports/2026-07-04-P2-investigation.md §修正案＋コミット `eceb9fd` の diff。
- 副次: `data/investers/` の更新停止12系列（dow, nickel 等・6/8停止）の扱い
  （update_external の対象化 or パネル分離）は別途判断。鮮度ガードにより
  実害は既に遮断済み＝急ぎではない。
- 判断: ＜未記入＞


## Resolved

### P-1 [config] monitor_config.json の認定値の凍結（2026-07-04 解決）
- 判断: **ユーザー承認「残りはすべて推奨の内容で決定」（2026-07-04）**＝
  claimed_sr/tol は seed どおり（combo 0.45 / eq 0.45 / ts 0.60）・combo の
  sigma_ann_max のみ 0.12→0.20 に修正して全系列 CONFIRMED 凍結。
  e過程 wealth は全て初期値1.0＝リセット不要を確認。以後の変更禁止。

### P-3 [approve] 週次 K 上限の設定確認（2026-07-04 解決）
- 判断: **ユーザー承認（同上）**＝ k_weekly_limit=12（週≈3サイクル）で確定。変更なし。

### P-2 / P-2b [investigate→approve] Phase 2 成果物間の乖離 → 月次会計バグ修正（2026-07-04 解決）
- 根本原因: `DataFrame.asof` の全列非NaN巻き戻り（詳細: ops_review_reports/2026-07-04-P2-investigation.md）。
- 判断: **ユーザー承認「P-2b 承認、修正して」（2026-07-04）** → 修正適用（コミット `eceb9fd`）。
- 検証: months⇔equity_daily 完全一致（combo −4.09%）・ops_review 初の全クリーン
  （FAIL=0/WARN=0）・関連テスト18本緑・loop_lint L1 は承認済みコミットとして除外登録。
- 残件は P-4（ops リポ側の適用）へ分離。
