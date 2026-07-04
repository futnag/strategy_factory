---
name: data-qa
description: ローカル市場データ（data/）の品質監査を実行し、findings の解釈と修復提案を行う。バー不変条件・カレンダー/ミラー整合・調整係数照合・リステートメント検知・スキーマドリフト。夜次（既定）と月次 --deep。flag, don't clean＝自動修復はしない。
---

# /data-qa — データ品質監査（1起動＝1監査＋解釈レポート）

決定論的コアは `examples/data_qa.py`。**数値・判定はスクリプトが計算し、あなた（LLM）は
解釈・優先順位付け・修復提案のみ**を行う。

## ガードレール（違反不可）

1. **flag, don't clean**: data/ 配下の市場データを直接編集・削除・「修正」しない。
   修復は必ず既存の再生成経路（`store.materialize_wide()`・`store.rebuild_adjusted()`・
   ミラー再取得スクリプト）の**提案**として人間に提示（軽微なものは提案の上で実行可＝下記）。
2. 書き込み先: `research_ops/`（ログ・レポート・契約）と `data/qa/`（ハッシュ台帳）のみ。
3. リステートメント検知が **事前登録済み結果に影響し得るファイル変更**を報告した場合、
   必ず research_loop/LESSONS.md への記録を提案する（判定の再現性に関わる一級事象）。
4. スキーマ契約（research_ops/data_contract.json）の更新（--init-contract）は、
   ドリフトが「意図した仕様変更」と確認できた場合のみ。
5. git コミットは1回（research_ops/ の更新分）。

## 手順

### Step 1 実行
```powershell
# 夜次（既定・直近45営業日窓）
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\data_qa.py
# 月次 deep（全期間・全ファイルハッシュ・上場廃止監査）
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\data_qa.py --deep
```

### Step 2 解釈（severity 別の意味と定石対応）
- **FAIL**（構造破壊）: バー不変条件違反・調整再計算乖離・重複日付・契約列の欠落。
  → 原因特定が最優先。調整乖離は `rebuild_adjusted()` 提案・不変条件違反はミラー再取得提案。
- **WARN**（要判断）:
  - 「ミラーにあるが wide 未反映」→ materialize 遅延。`store.materialize_wide()` の実行を
    提案（低リスク・承認あれば実行可）。
  - 「|生リターン|>40% かつ factor=1」→ 数銘柄をサンプルで目視（分割見逃し vs 実際の暴騰落）。
    Web で当該銘柄のコーポレートアクションを確認してよい。
  - リステートメント検知 → 変更ファイルが過去の判定（docs/NN §5）の入力なら影響評価＋
    LESSONS 記録提案。
- **INFO**: 記録のみ。

### Step 3 レポートとコミット
- 要約＋解釈＋修復提案を `research_ops/data_qa_reports/YYYY-MM-DD.md` に書く
  （深刻度順・実行した修復があれば結果も）。
- `git add research_ops/` → `ops(data-qa): <mode> — FAIL n/WARN m と1行要約`。
- ユーザー向け1画面サマリ: FAIL/WARN の内訳・最重要 finding・提案アクション・
  実行済み修復。

## 運用メモ
- 推奨頻度: nightly は夜間パイプライン後（当面は手動・週1目安）・deep は月1。
- 初回や仕様変更後は `--init-contract` で契約を再固定（Step 2 の確認後のみ）。
