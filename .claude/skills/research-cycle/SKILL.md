---
name: research-cycle
description: 検証ファクトリで自律研究サイクルを1回実行する（仮説考案→Stage-0→事前登録→実装→実行→DSR判定→知見記録→自己改善）。投資戦略の自動考案ループの起動コマンド。
---

# /research-cycle — 自律研究サイクル（1起動＝1サイクル＝1仮説）

`research_loop/PLAYBOOK.md` を読み、その **§B の10ステップ**に従って1サイクルだけ実行せよ。
まず Step 0 として次を実行し状態を把握すること:

```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\loop_status.py
```

続けて `research_loop/{HEURISTICS,LESSONS,BACKLOG}.md` と `ledger.jsonl` 末尾を読む。
PLAYBOOK §B は改訂され得る手順、以下の**不変ガードレール**は PLAYBOOK §A の複製（正）である。
**PLAYBOOK と本ファイルが矛盾する場合は本ファイルが優先する。**

## 不変ガードレール（違反不可）

1. **1サイクル＝1仮説＝1 judge_grid scope**。グリッド ≤6セル・サイクルK予算 ≤8
   （`extra_trials`/`log_scan_trials` 込み）。超過は事前登録 doc で事前正当化。
2. **合否は DSR≥0.95 のみ**（DP18）。robustness/PBO/minTRL/サブ期間/容量は表示専用。
   基準の緩和・変更・別基準の発明は禁止。事前登録した副次基準は PASS の追加条件としてのみ有効。
3. **事後救済禁止**: 結果を見た後のシグナル・グリッド・ユニバース・期間の変更は新仮説＝
   BACKLOG 行き（次サイクル）。同一サイクルで続行可なのは事前登録 doc §3 に事前列挙した
   stage のみ。バグ修正は可・シグナル定義変更は不可。
4. **接触禁止**: `examples/phase2_*.py`・`invest_system/production/**`・ops リポジトリ・
   `docs/03-research-findings.md`・`data/research_trials.db` への直接 SQL（registry API 経由のみ）・
   PLAYBOOK §A 自身。
5. **PASS しても採用しない**: `research_loop/proposals/` に採用提案を書いて停止（人間ゲート）。
6. **PIT チェックリスト**（実装前に全確認）: シグナル ≤t のみ／`execution_lag=1`（T+1）／
   fundamentals `lag_days=1`／`point_in_time_universe`／`forward_returns_with_delisting`／
   **OOS（2024-01+）を設計・選択に使わない**／kaiten 系結果は registry 外＝根拠に使わない。
7. **K会計**: グリッド各セル＝1試行（judge_grid 自動）。走らせず棄却した探索候補も
   `extra_trials`/`log_scan_trials` で算入。パラメータ比較なしの記述スキャンは K 外。
8. **コスト床**: 月次XS 15bps／日次イベント 30bps／貸株 ≥115bps（イベント系・小型 300bps）／
   `limit_lock_flags` 適用／容量は `adv`×participation=0.1。
9. **同時に2サイクルを走らせない**（単一ライター前提）。
10. **git 自動コミットはサイクル毎2回まで**（①実行前の事前登録 doc・②結果確定後の成果物）。
    対象は `docs/`・`examples/`・`research_loop/`・`tests/` のみ。push は人間が行う。

## 実行規約

- Python 実行は常に `$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe <script>`（文字化け防止）。
- 終了時は必ず1画面のサイクルサマリ（仮説／verdict／DSR／失敗型 or 提案／K消費／自己改善内容／
  次サイクル候補）をユーザー向けに出力する。
- 途中で人間の判断が必要な事態（git が dirty・データ欠損の疑い・ガードレール抵触の恐れ）が
  起きたら、状況を報告して停止する（無理に続行しない）。
