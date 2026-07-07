---
name: sairyo-retro
description: 裁量レーン（market/us/jp-open/news/kaigai/kabu系/kessan/tactical/pretrade）の月次振り返り。jp-open 予想のキャリブレーション（的中率・confidence別の過信検査）、tactical-brief レーンB提案の追跡（実行/棄却/滞留）、kabu系レポートの監視KPI答え合わせ率、スキル別利用実績と削除候補の棚卸しを行い、裁量レーン全体を「記録なき主張は無効」の規律で監査する。/loop-retro の裁量レーン版。読み取り＋output/sairyo/ への書き込みのみ。
---

# /sairyo-retro — 裁量レーンの月次振り返り（1起動＝1ヶ月分の監査）

裁量スキル群は「出力を出す」側に偏っている。**答え合わせと淘汰**がこのスキルの仕事。
的中しないこと自体は失敗ではない（方向予測不能は確定知見）が、**記録の欠落と滞留は失敗**。

## ガードレール（違反不可）

1. 書き込みは `output/sairyo/` のみ。スキル定義（`.claude/skills/`）・CLAUDE.md の改廃は
   **提案止まり**（実施は人間承認後の別作業）。
2. 研究レーンに書かない。裁量観察から研究候補が出たら「/research-scout に渡す候補」として
   列挙するだけ（IDEAS へは書かない）。
3. 数値の集計はインライン python で行い、計算式をレポートに残す（再現可能に）。
   集計ロジックが安定したら `examples/sairyo_retro.py` への昇格を提案してよい。
4. 結果が人間の頭の中にしかない項目（レーンB を実際に執行したか等）は勝手に推定せず、
   「要記入」欄としてユーザーに返す。前回レポートのユーザー記入があれば取り込む。

## 手順

### Step 1 対象期間と材料
対象は前回 retro 以降（初回は全期間）。材料は `output/` 以下すべて:
`jp_open/predictions.jsonl`・`jp_open/brief_*`・`tactical/brief_*`・`kabu/*`（dd/funda/screen/
kessan）・`news/scan_*`・`kaigai/scan_*`・`market_brief/*.json`・`us_brief/*.json`・
`phase2_pretrade/*`・`phase2_triage/*`・前回の `sairyo/retro_*.md`（ユーザー記入含む）。

### Step 2 集計（インライン python・式を残す）
1. **利用実績**: スキル別の出力ファイル数・最終使用日（output/ サブディレクトリの
   ファイル日付を代理指標に）。**90日以上未使用は削除候補**として明示。
2. **jp-open キャリブレーション**: predictions.jsonl から寄付/日中の的中率（up/flat/down 一致）、
   confidence 別（medium が low を下回れば過信＝要報告）、realized 未記入率（記録規律の監査）。
   n<10 なら「評価保留（サンプル不足）」とだけ書く（少数で結論を出さない）。
3. **レーンB台帳**: tactical-brief 各回のレーンB提案を抽出し、状態（新規/実行/棄却/滞留/
   結果要記入）で分類。**3回以上滞留した案の決断**をユーザーに催促。
4. **KPI 答え合わせ率**: kabu-dd/funda レポートの監視KPIリストのうち、再訪（後続レポートや
   kessan-preview）で実際に確認された割合。
5. **鮮度・運用**: market/us brief の実行間隔、phase2-triage の失敗パターン頻度。

### Step 3 解釈と提案（LLM の仕事）
- **続ける/減らす/やめる**をスキル単位で提案（利用実績×成果物が判断に使われた形跡）。
- jp-open: 過信の有無と、的中率が偶然域に留まる場合の一言（知見との整合＝想定通り）。
- レーンB: 滞留案の decision 催促・実行済み案の教訓1行。
- 研究レーンへの橋: 裁量観察のうち検証可能な形にできそうなもの（あれば。無ければ無しと書く）。
- 記録規律の違反（predictions 未記入・3点セット欠落のレーンB案 等）の指摘。

### Step 4 出力
`output/sairyo/retro_{YYYY-MM}.md`:
①利用実績表と削除候補 ②jp-open キャリブレーション ③レーンB台帳（要記入欄付き）
④KPI答え合わせ率 ⑤提案（改廃・頻度・決断催促） ⑥ユーザー記入欄（次回 retro が読む）。
ユーザーへは「最重要の提案3つ＋要記入事項」を1画面で。

## 運用メモ
- 推奨頻度: 月次（/ops-review・/strategy-monitor と同時期にまとめて）。
- このスキル自身も対象（retro が使われなくなったら裁量レーン全体の見直し時）。
- 初回は基準線づくり＝判定を急がない（利用0のスキルも「作った月」は削除候補にしない）。
