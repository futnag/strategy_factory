---
name: data-update
description: ローカル市場データ（data/）を全ソース一気通貫で最新化する（J-Quants差分→外部価格→TDnet/ToSTNeT/EDINET→派生再生成→品質監査→報告）。既存の差分更新スクリプト群の司令塔＝新規取得ロジックは持たない。ToSTNeT は約2週間で消えるため優先。
---

# /data-update — ローカルデータの一括最新化（1起動＝1更新サイクル）

既存の冪等・差分更新スクリプト群を**正しい順序**で実行し、最後に品質監査まで通す司令塔。
取得ロジックはここに書かない（各 update_* スクリプトが正）。

## ガードレール（違反不可）

1. 各ソースの取得は**必ず既存スクリプト経由**（API 直叩き・手パッチ禁止）。失敗した
   ソースはスキップして続行し、最後にまとめて報告（1ソースの失敗で全体を止めない）。
2. J-Quants のレート制限を尊重（`J_QUANTS_MIN_INTERVAL` は .env の値を使う・下げない）。
   **全量再取得（download_jquants --all）はこのスキルでは実行しない**（README どおり
   手元ターミナル推奨＝提案のみ）。
3. 市場データの手編集禁止。派生の再生成は正規経路（materialize/rebuild_adjusted）のみ。
4. 書き込み: `data/`（スクリプト経由）・`research_ops/`（ログ）。コミットは1回
   （research_ops のみ＝data/ は gitignore）。

## 手順（実行順序が重要）

### Step 1 更新前スナップショット
`loop_status.py` の鮮度セクション（または data_qa の calendar 所見）で現状を把握。

### Step 2 ソース更新（順序固定・各コマンドは失敗しても次へ）
```powershell
# 1) 消えるものから: ToSTNeT（公表ページは約2週間で消滅）
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_tostnet.py
# 2) TDnet 適時開示一覧（公表1ヶ月窓）
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_tdnet.py
# 3) J-Quants 差分（欠損日のみ・既定で Silver 増分 materialize 込み）
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_data.py
# 4) 外部11資産（検証付き追記）
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_external.py
# 5) EDINET 書類一覧（差分）
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\download_edinet.py --list-only 2>$null  # 引数は先頭docstringで要確認
# 6) JSF（手動DL前提のソースは状態確認のみ）・JPX ウェイトCSV refresh
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_jpx_topix_lists.py --refresh
```
各スクリプトの正確な引数は**先頭 docstring を必ず確認**してから実行（引数を推測しない）。

### Step 3 派生の再生成判定
- update_data.py が materialize 済みか確認（--no-materialize で走らせた場合や
  取得だけ成功して materialize が古い場合は `store.materialize_wide()`）。
- adj_factor に変化があった場合のみ `store.rebuild_adjusted()` を提案・実行。
- features（Gold）の再計算は重い＝必要になった研究サイクル側で行う（ここではしない）。

### Step 4 品質監査（更新の検収）
`/data-qa` の nightly 相当を実行:
```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\data_qa.py
```
更新前に出ていた calendar/鮮度系の WARN が解消したかを必ず突合。
（リステートメント検知が発火した場合は data-qa の手順に従い影響評価。）

### Step 5 報告とコミット
- `research_ops/data_update_log.jsonl` に1行 append（各ソースの 成功/失敗/追加日数・
  materialize 有無・QA 結果の FAIL/WARN 数）。
- `git add research_ops/` → `ops(data-update): <期間> — <ソース別1行要約>`。
- サマリ: 更新されたソースと日数・失敗と原因・QA 差分・次回への注意。

## 運用メモ
- 推奨頻度: 週1〜2（ToSTNeT の2週間窓が律速）。夜間 ops（GitHub Actions）とは独立＝
  こちらは**ローカル研究環境**の鮮度維持。
- 大規模欠損（数週間以上）を検知したら、全量系は手元ターミナルでの実行を提案する。
