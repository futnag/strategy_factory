# PENDING — 人間の承認/判断 待ち行列

**運用ルール**: `P-n` 番号制。スキル/オペレータが append し、人間が判断を追記して
`resolved` にする（削除しない）。**open の項目に依存するアクションは実行しない**。
種別: `approve`（PASS採用・platform変更・ToSグレー取得）/ `investigate`（要調査）/
`config`（凍結値の確認）/ `incident`（HARD違反・障害）。

---

## Open

### P-6 [config] 週次 K 上限の初週拡大＝要・復帰判断（2026-07-04・人間設定）
- **人間が k_weekly_limit を 12→48 に拡大**（初週の稼働テスト目的・研究サイクル経路を
  実際に回すため）。loop_lint 確認済み＝32/48・全ガードレール通過（BUDGET 解除）。
- **注意**: 48/週 ≈ 12サイクル/週は多重検定規律（anti-p-hacking スロットル）を4倍緩める。
  **初週テスト用の一時措置**であり、通常運用（週≈3サイクル）に戻すなら 12 へ復帰。
- 復帰の目安: 稼働テスト完了時（〜2026-07-11）。維持するなら本項に「維持」と記載。
- 判断: ＜未記入・初週テスト中＞

## Resolved

### P-5 / P-5b [investigate→approve] 夜間 gate の恒常 false（キルスイッチ約18日無効）→ 修正・稼働確認（2026-07-04 解決）
- 根本原因: Data completeness gate の `missing==0` が J-Quants 公表ラグと衝突し
  ready=false 固定化＝reconcile/dashboard/commit/キルスイッチが 6/17 以降毎晩 skip。
- 判断: **ユーザー承認「P-5b 承認、push して」（2026-07-04）**。
- 実施: gate を per-dataset ラグ許容判定へ（ops `be1a4d1`）＋欠落した
  `up = DataUpdater()` の回帰修正（`06bdee0`・手動起動で NameError を検出し即修正）。
- **稼働確認**: workflow_dispatch 再実行が 16m50s で成功＝gate 通過・reconcile・
  Supabase push・dashboard build・**キルスイッチ評価**が全て実行。bot 自動コミット
  `aeafdeb data: nightly 2026-07-04`＝data.json 自動更新復活（cum_net −4.12% /
  asof 2026-07-03 / kill OK）。alert issue の誤発火なし。
- 教訓: YAML 埋め込み Python はローカルで「書き直したコピー」でなく**ファイルから抽出して
  実行**して検証すべき（今回 up= 欠落を初回 push で見逃した）。

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
