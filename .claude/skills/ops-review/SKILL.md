---
name: ops-review
description: Phase 2 運用のレビュー（月次/リバランス毎）。成果物間リコンサイル・執行整合（fill⇔T+1寄付）・コスト監査（Carver speed limit）・月次アトリビューション。数値はコードが計算・LLM は解釈と提案のみ。Phase 2 への操作は一切しない。
---

# /ops-review — 運用レビュー（1起動＝1レビュー＋レポート）

決定論的コアは `examples/ops_review.py`（対象月は既定で最新・`--month YYYY-MM` 指定可）。

## ガードレール（違反不可）

1. **Phase 2 系（examples/phase2_*.py・data/phase2/ の成果物・ops リポ）への書き込み・
   修復・発注は一切禁止**。調査は読み取りのみ可（phase2_reconcile.py 等のコードを読んで
   成果物のセマンティクスを確認するのは可）。
2. リコンサイル FAIL は**最優先で扱う**（ペーパー運用では成果物間のサイレント・ドリフトが
   実弾のブローカー・ブレイク相当）。ただしエスカレーション前に「定義差」の可能性を
   コード読解で切り分けること（例: カーブの列とmonths.csv の列が別の量を測っている等）。
3. 書き込み先: `research_ops/` のみ。git コミットは1回。

## 手順

### Step 1 実行
```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\ops_review.py
```

### Step 2 解釈（チェック別の定石）
- **reconcile.curve_vs_status FAIL**: まず phase2_reconcile.py と各成果物の生成コードを
  **読んで**、比較対象が同じ量か確認（基準日・費用込み/抜き・ヘッジ込み/抜き）。
  定義差なら ops_review.py の比較を修正する提案（このスキルの管轄）・真のドリフトなら
  人間へエスカレーション（ops リポの夜次ジョブ確認を提案）。
- **execution.fill_vs_open**: 乖離>10bp はペーパー執行規約（T+1寄付）の破れ＝
  fills 生成ロジックか始値データの問題。/data-qa と突合。
- **cost.speed_limit WARN**: 回転かコスト前提の見直しを研究側（H-3）に接続。
- **attribution**: 実現ボラ vs 目標の乖離（TSMOM 10%目標）・combo 再構成差の拡大傾向を
  時系列で監視（ログの過去行と比較）。
- 実弾移行後: fills_actual が入ると execution チェックは実測スリッページ分解
  （遅延/スプレッド/執行）に自然拡張される（pysystemtrade 方式・スクリプト拡張の提案可）。

### Step 3 レポートとコミット
- `research_ops/ops_review_reports/YYYY-MM.md`: 判定表・前回比・調査結果（定義差 or 実障害）・
  提案。月末リバランス直後の回は注文プレビュー整合（orders vs intended）を重点。
- `git add research_ops/` → `ops(review): YYYY-MM — <最重要所見1行>`。
- サマリ: FAIL/WARN・原因切り分け結果・人間アクション要否。

## 運用メモ
- 推奨頻度: 月次（リバランス＋reconcile 後）＋ /strategy-monitor と同時運用が自然
  （monitor=統計的健全性・review=会計的/執行的整合）。
- 年次: Carver テンプレート（コスト分解・ベンチ比較・ルール別寄与）での年間レビューを
  このスキルの拡張として実施（ログ12ヶ月分が入力）。
