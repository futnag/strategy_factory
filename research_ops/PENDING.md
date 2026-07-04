# PENDING — 人間の承認/判断 待ち行列

**運用ルール**: `P-n` 番号制。スキル/オペレータが append し、人間が判断を追記して
`resolved` にする（削除しない）。**open の項目に依存するアクションは実行しない**。
種別: `approve`（PASS採用・platform変更・ToSグレー取得）/ `investigate`（要調査）/
`config`（凍結値の確認）/ `incident`（HARD違反・障害）。

---

## Open

### P-5b [approve] gate 修正の push 承認（2026-07-04・調査完了に伴う後継）
- **調査完了・根本原因確定**: 夜間は毎晩**成功**していたが、Data completeness gate が
  `missing==0` を要求＝J-Quants 公表ラグ（直近営業日は当日未公表で daily_quotes が
  恒常的に1日欠損）で **ready=false が固定化**（6/28・7/1・7/3 いずれも missing=1）。
  結果、reconcile・dashboard build・commit・**キルスイッチ評価**が 6/17 以降**毎晩 skip**。
  ＝自動キルスイッチが約18日間無効だった（安全上の問題。現 DD −4.09% は閾値内だが
  下落時に自動警報が出ない状態だった）。
- 実施済み: ops リポ phase2.yml の gate を per-dataset 判定へ修正（LAG_TOLERANCE_BD=3・
  シード未完と定常ラグを分離）。**ローカルコミット `be1a4d1`・未 push**（本番キルスイッチの
  発火条件を変えるため push 前にレビュー用に保留）。3シナリオで新旧比較検証済み。
- 承認事項: `be1a4d1` の push 可否。push すれば当夜（21:30 JST）から reconcile と
  キルスイッチ評価が復活し data.json も自動更新される。
- 判断: ＜未記入＞

## Resolved

### P-4 [approve] ops リポへの同修正の適用（2026-07-04 解決）
- **調査結果: ops リポにコード修正は不要**。phase2.yml は strategy_factory を
  `ref:` 指定なしで checkout（`main` HEAD を使用）＝main を push すれば次回夜間から
  修正版 reconcile が自動適用される。
- 実施: 配信中の `data.json` が 6/17 以来 cum_net +0.61%（誤）を表示していたため、
  ローカルの修正済み成果物から再生成して ops リポにコミット（`99b4cc8`・要 push）＝
  正値 cum_net −4.09% / asof 2026-07-03 / kill OK に即時是正。
- 副次: `data/investers/` の更新停止12系列は鮮度ガードで実害遮断済み＝別途判断（急がず）。
- **push 待ち**: main（sf・`eceb9fd` 含む）＋ ops（`99b4cc8`）の両方。夜間ワークフロー
  停止疑いは P-5 へ分離。

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
