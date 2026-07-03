# research_loop — 自律研究サイクル（自己改善型・戦略考案ループ）

Claude Code の自律セッションが、検証ファクトリの規律（PIT・事前登録・永続デフレート DSR）の
**内側**で「仮説考案 → 検証 → 判定 → 知見記録 → 自己改善」を回すための状態ディレクトリ。

## 起動

```
/research-cycle
```

1回の起動＝1サイクル（1仮説）。セッションは `PLAYBOOK.md` の手順に従い、
終了時にサイクルサマリを出力する。**PASS しても自動採用はしない**（`proposals/` に提案を書いて
停止＝人間ゲート）。Phase 2 本番系・`docs/03`（知見の正本）には触れない。

## ファイルマップ

| ファイル | 役割 | 書き込み規則 |
|---|---|---|
| `PLAYBOOK.md` | サイクル手順書 | §A 不変（人間のみ）／§B ループが改訂可（§C に履歴） |
| `HEURISTICS.md` | 仮説生成 prior＋Stage-0 キルチェック | H-n append-only・証拠引用必須 |
| `LESSONS.md` | 失敗型タクソノミー＋蓄積知見 | append-only。人間が定期的に docs/03 へ編入 |
| `BACKLOG.md` | 仮説キュー＋除外リスト＋Stage-0 棄却台帳 | 状態更新＋追記 |
| `ledger.jsonl` | サイクル毎のファネル指標 | 1行/サイクル append-only |
| `proposals/` | PASS 時の採用提案（人間レビュー待ち） | サイクルからは書くのみ |

状態把握は `examples/loop_status.py`（レジストリ scope 表・直近試行・キュー・台帳・データ鮮度）。

## ledger.jsonl スキーマ

```json
{"cycle_id":"2026-07-04-margin-alert","date":"2026-07-04","domain":"event",
 "scope":"margin_alert_event","hypothesis_short":"日々公表/規制指定の前後ドリフト",
 "funnel":{"proposed":3,"stage0_killed":2,"preregistered":1,"run":1},
 "verdict":"FAIL","best_dsr":0.41,"k_added":4,"failure_type":"F8 cost死",
 "artifacts":{"prereg":"docs/53-margin-alert-preregistration.md",
              "script":"examples/research_margin_alert.py",
              "report":"data/reports/margin_alert_event.html"},
 "wall_minutes":95,"amendments":["H-12 added"],"notes":""}
```

`verdict` は `PASS | FAIL | no_viable`。`domain` は `equity_xs | event | ts_multi | crypto`。
`failure_type` は LESSONS §1 の F1〜F9（PASS/no_viable 時は null）。
