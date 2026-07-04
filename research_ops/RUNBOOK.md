# RUNBOOK — 無人運用の開始・運転・介入手順

対象: strategy_factory の自律研究/運用ループ（16スキル＋機械強制ゲート）。
本書は「人間がやること」の全てを1ファイルにまとめた運転マニュアル。

---

## 0. 全体像（1分で）

```
夜間: タスクスケジューラ → claude -p "/operator"（Opus）
        └ operator: ロック → loop_lint（機械ゲート）→ 優先規則で1アクション
            ├ データ系: /data-update → /data-qa（検収）
            ├ 月初:    /strategy-monitor → /ops-review
            ├ 研究系:  /red-team → /research-cycle（週次K予算内のみ）
            ├ 供給系:  /research-scout・/data-scout・/data-acquire
            └ ブリーフ: research_ops/briefs/YYYY-MM-DD.md
朝（人間・3分）: briefs/ の1ページ → PENDING.md の open に判断を書く
週次（人間・10分）: git push・loop_lint の BUDGET 消化確認
```

規律の担保は「モデルの善意」ではなく: loop_lint（§A不変・prereg先行コミット・K予算・
接触禁止パス）＋ registry（DSR/K会計）＋ PENDING（人間ゲート）。

## 1. 開始前チェックリスト（一度だけ）

- [ ] **P-1 監視の認定値を凍結**: `research_ops/monitor_config.json` の各 `claimed_sr_ann` /
      `sigma_ann_max` / `tol_sr_ann` を docs/03 §6.15-6.16 と突合し、納得したら
      `"status": "CONFIRMED"`・`"frozen_by": "<あなた>"` に書き換え（以後変更禁止＝攻撃面）。
- [ ] **P-2 Phase 2 乖離の解決**: equity_daily と status/months の符号乖離（PENDING 参照）。
      Claude に「P-2 を調査して」で読み取り専用調査を依頼可。**未解決のまま無人化すると
      monitor/ops-review の読みが信用できない**ため、開始前に必須。
- [ ] **P-3 週次 K 上限の確認**: `research_ops/loop_lint_baseline.json` の
      `k_weekly_limit: 12`（≈週3サイクル）。変更するならここを編集（人間のみ）。
- [ ] **git push**: ローカルコミットを origin へ（push は常に人間の操作）。
- [ ] **registry バックアップ**: `data/research_trials.db` を `data/backup/` へコピー
      （以後 月1・PLAYBOOK Step 10 の運用ノート）。
- [ ] **権限設定**: 下記 §2 の方式を選択。
- [ ] **フルテスト**: `.\.venv\Scripts\python.exe -m pytest -q` が全緑であること。

## 2. 権限方式の選択（無人実行の要）

headless（`claude -p`）は権限プロンプトに答える人がいない。選択肢:

**方式A（推奨・簡便）: bypassPermissions**
```
claude -p "/operator" --model opus --permission-mode bypassPermissions
```
- リスクと緩和: 全ツールが無確認になるが、(1) loop_lint が接触禁止パス・§A・K予算を
  機械強制 (2) push はスキル側で禁止 (3) 取得系は ToS 遵守を SKILL に明記 (4) 作業対象は
  この repo のみ。**このリポジトリ専用の運用として許容範囲**（他ディレクトリで使い回さない）。

**方式B（保守的）: allowlist 整備**
- 対話セッションで `/fewer-permission-prompts` を実行し、頻出コマンドを
  `.claude/settings.local.json` に許可登録 → `--permission-mode acceptEdits` で運用。
- 初期は未登録コマンドで停止が起きるため、1-2週は方式Bで様子見→Aへ、も可。

## 3. スケジューラ登録（Windows）

`research_ops/run_operator.cmd`（同梱・下記）を使う:

```bat
schtasks /create /tn "strategy-factory-operator" ^
  /tr "C:\Users\futos\claude_local_sessions\research_ops\run_operator.cmd" ^
  /sc weekly /d MON,TUE,WED,THU,FRI /st 22:30 /f
```

- PC がスリープする環境なら、タスクのプロパティで「タスク実行のためにスリープ解除」を ON。
- 停止（無効化）: `schtasks /change /tn "strategy-factory-operator" /disable`
- ログ: `research_ops/logs/operator_YYYYMMDD.log`（cmd が追記）。
- 注意: `claude` の CLI フラグはバージョンで変わり得る＝初回は
  `claude --help` で `-p` / `--model` / `--permission-mode` を確認してから登録。

## 4. Opus に渡すプロンプト集

**夜間の定期実行（スケジューラが送る・これが基本）**
```
/operator
```
それ以外の文言は不要（SKILL.md が全手順を持つ）。モデルは `--model opus`。

**判断だけ見たい（実行しない・手動確認用）**
```
/operator plan
```

**個別スキルを手動で回す時（対話セッション）** — スキル名だけで完結:
```
/research-cycle          ← BACKLOG 先頭 ⬜ を1サイクル（設計系＝できれば強いモデルで）
/research-scout [テーマ]  ← 文献調査（例: /research-scout クリプト マイクロストラクチャ）
/red-team docs/60-xxx.md ← 事前登録草稿の敵対的レビュー
/data-update             ← ローカルデータ一括更新＋QA検収
/data-acquire            ← 調達待ち行列の先頭を実行（D-n 指定可）
/strategy-monitor        ← 稼働戦略の健全性判定（月次）
/ops-review              ← 運用リコンサイル・執行整合（月次）
/loop-retro              ← ループ自体の振り返り（四半期）
```

**モデルの使い分け（operator の SKILL にも記載）**
| 用途 | モデル |
|---|---|
| operator・data-update/qa/acquire・monitor・ops-review | Opus / 安価モデルで十分 |
| research-cycle の設計（Step1-3）・scout の統合判断・red-team | そのとき使える最強モデル推奨 |
| 迷ったら | Opus で開始（lint と red-team 必須化が下限品質を担保） |

**Opus セッションに追加コンテキストは渡さない**こと。スキル＋research_loop/・research_ops/
の状態ファイルが全て。口頭の追加指示は規律の外側に立つため、恒常的な指示は
PLAYBOOK §B 改訂（cycle Step 8）か HEURISTICS 追加として残す。

## 5. 日々の人間の仕事

- **朝（3分）**: `research_ops/briefs/` の最新を読む。⚠ があれば `PENDING.md` の open に
  「判断: 」を書き込み `Resolved` へ移す。
- **週次（10分）**: `git log --oneline` 確認 → `git push`。lint の週次 K 消化を確認
  （ブリーフに出る）。
- **月次**: monitor/ops-review レポート・registry バックアップ・（あれば）PASS 提案
  （`research_loop/proposals/`）のレビュー。
- **四半期**: /loop-retro のレポート → operator 閾値・探索方向の再配分を判断。

## 6. 介入・停止・トラブルシュート

| 症状 | 対応 |
|---|---|
| 今すぐ全停止したい | `schtasks /change /tn "strategy-factory-operator" /disable`。実行中なら次のアクション境界で止まる（1セッション1アクション） |
| `operator.lock` が残っている | 6h 未満＝実行中（触らない）。6h 以上＝異常終了。中身を確認し削除。次回 lint が stale を報告 |
| loop_lint HARD（exit 1） | ループは自動停止済み。PENDING の incident を読み原因除去。**§A 変更が原因なら意図確認のうえ `--freeze-a`（人間のみ）** |
| BUDGET（exit 2）が続く | 正常（週次スロットル）。上げたいなら baseline の `k_weekly_limit` |
| セッション上限で中断 | 正常停止設計（ロック解放・ログ記録）。次回起動で自動再開。子エージェント再開は対話セッションで「<agent> を再開して」 |
| モデルが変な設計をした | red-team とテンプレートが一次防衛。すり抜けたら FAIL 後の LESSONS に「レビューの見逃し」として記録される設計＝次サイクルで改善 |
| 全部やり直したい | git なので任意コミットに戻せる。registry（.db）だけは append-only ＝ data/backup から復元 |

## 7. 最初の2週間に起きること（期待値）

1週目: 週次 K が人力ビルド分で消化済み → operator はデータ系（D-2 取得・鮮度維持・QA）中心。
2週目〜: `topix_staged_flow`（red-team 修正済み設計）から研究サイクル再開・週3ペース。
月初: monitor＋ops-review が走り、Phase 2 の統計的/会計的健全性レポートがブリーフに載る。
FAIL が続くのは正常（基本率）。見るべきは「学習密度（H/サイクル）」と PENDING の質。
