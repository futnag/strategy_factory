---
name: strategy-monitor
description: 稼働戦略（Phase 2 ペーパー/実弾スリーブ）の健全性を月次判定する。PSR/MinTRL・FJW e過程（anytime-valid 減衰検知）・CUSUM・キルスイッチ閾値距離。非対称キルルール＝負の証拠でのみエスカレーション。判定はコードが計算・LLM は解釈とレポートのみ。
---

# /strategy-monitor — 稼働戦略の健全性判定（1起動＝1判定＋レポート）

決定論的コアは `examples/strategy_monitor.py`。帰無仮説（認定値）は
`research_ops/monitor_config.json` に**人間が凍結**したもの。

## ガードレール（違反不可）

1. **認定値（m・tol・σ_max）の事後調整禁止**（攻撃面＝config は人間のみ編集可。
   TO_CONFIRM 状態なら毎回その旨を報告する）。
2. **e過程・CUSUM の走行状態（monitor_state.json）をリセット・改変しない**
   （anytime-valid 性が壊れる）。改善検知を測りたい場合は別の e 過程を新設提案。
3. **非対称キルルール**: エスカレーション（amber/review/kill_candidate）は負の証拠
   （e値・CUSUM・DD閾値）でのみ。「まだ有意に正でない」は insufficient_evidence のまま
   ＝10年フラットでも「有意に負」にならなければ殺さない（Carver の教訓）。
4. **kill_candidate が出ても実行しない**: Phase 2 への操作は一切禁止（§不変ガードレール
   と同じ人間ゲート）。レポートで提案するのみ。
5. 書き込み先: `research_ops/` のみ。git コミットは1回。

## 手順

### Step 1 実行
```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\strategy_monitor.py
```

### Step 2 解釈（判定の意味）
- **insufficient_evidence**: MinTRL 未達＝正直な既定状態。1〜3年はこれが正常。
  レポートでは充足率の推移だけ示す（無理に結論を出さない）。
- **amber**（e≥5 or CUSUM 発火 or DD alert）: 監視強化。単月ノイズか systematic かを
  当月のスリーブ別リターン・status.json・直近の市場環境で文脈化する。
- **review**（e≥10 or DD derisk）: 正式レビュー提案＝docs/02 D5 のデリスク手順の
  発動を人間に推奨し、根拠（e値の由来月・CUSUM の蓄積経路）を明示。
- **kill_candidate**（e≥20 or DD stop）: α=5% 相当の統計的減衰証拠 or キルスイッチ域。
  停止提案＋事後分析（decay か regime か執行かの切り分け計画）をレポートに含める。

### Step 3 レポートとコミット
- `research_ops/monitor_reports/YYYY-MM.md` に: 系列別判定表・前回からの変化・
  e値/CUSUM の推移解釈・（あれば）提案アクション。
- `git add research_ops/` → `ops(monitor): YYYY-MM — <最重要判定の1行>`。
- ユーザー向け1画面サマリ（判定・根拠・次回までの注視点）。

## 運用メモ
- 推奨頻度: 月次（月末リバランス＋reconcile 後）。amber 以上が出たら随時再実行可
  （anytime-valid なので覗き放題＝それが e 過程の意義）。
- 認定値の初期 seed は保守側（計画帯下限）。人間が確認したら status を CONFIRMED に。
- 将来: 実弾移行時も同じ config/state で継続（ペーパー→実弾の接続点で e 過程は
  リセットせず、境界日を config に記録するだけ）。
