---
name: jp-open
description: 当日の日本株見通しブリーフ。/us-brief の japan_handoff（NK225先物・USDJPY・米指数）＋market-brief 状態＋当日イベントから、寄付ギャップ（先物由来＝ほぼ機械的）と日中シナリオ（不確実）を分離して提示。予想は output/jp_open/predictions.jsonl に必ず記録し、次回実行時に答え合わせ＝キャリブレーション自己監査。確定知見「方向予測に有意エッジ無し」を毎回明記＝売買シグナルではない。
---

# /jp-open — 当日の日本株ブリーフ（1起動＝1予想＋前回の答え合わせ）

## ガードレール（違反不可）

1. **売買シグナルではない**（冒頭に定型注記を必ず入れる）:
   「本 repo の確定知見: 単一資産の方向予測に有意エッジ無し（docs/03）。本ブリーフは
   市況理解とキャリブレーション記録が目的。Phase 2 の執行はルールが決める。」
2. **記録なき予想の禁止**: 見通しを出したら必ず predictions.jsonl に1行 append する。
   答え合わせと的中率の提示が本スキルの本体機能（当たること、ではない）。
3. **2層分離**: 寄付ギャップ（先物とほぼ裁定＝機械的推計）と日中方向（ほぼ不可知＝
   シナリオ形式）を混ぜない。confidence は low / medium のみ（high 禁止＝過信ガード）。
4. 裁量レーン: 書き込みは `output/jp_open/` のみ。研究・レジストリ・Phase 2 に触れない。

## 手順

### Step 1 入力集め
- 当日の `output/us_brief/brief_{YYYYMMDD}.json`。**無ければ先に /us-brief を実行**。
- 直近の `output/market_brief/brief_*.json`（あれば。セクターRS・需給の文脈用）。
- Web: CME/大取ナイトの日経先物現値（JSON の implied はあくまで日足近似）、寄付前の
  外資系注文動向の報道、当日の国内イベント（主要決算・経済指標・日銀・SQ・指数イベント。
  決算は EDINET DB get_earnings_calendar やローカル `equities/events.py` でも補完可）。
- 休場日なら「本日休場」とだけ報告して終了（記録もしない）。

### Step 2 前回予想の答え合わせ
`output/jp_open/predictions.jsonl` の `realized` が空の行を、ローカルデータで埋める:
TOPIX は `data/jquants/indices/code_0000.parquet`、間に合わなければ `nk225_fut`
（`invest_system.data.external.load_external_prices`）で近似し、近似した旨を残す。
埋めたら直近20件で的中率（寄付方向・日中方向 各々）を集計して表示。
サンプルが偶然と区別できない水準に留まるならそれを明記（それ自体が知見の再確認）。

### Step 3 本日の見通し
1. **寄付**: implied_open_gap_pct と Web の先物現値から「ギャップ方向と規模（%目安）」。
2. **日中シナリオ**: メイン1本＋リスク1本（各に前提条件を1行。例:「米金利上昇が
   続けばグロース売り継続、ただし USDJPY 160 割れなら輸出主導で失速」）。
3. **セクター注目**: US→JP 対応（NASDAQ/SOX→電機・半導体、米金利→銀行・グロース、
   WTI→石油/商社、USDJPY→輸出/内需、金・銅→非鉄）を market-brief の RS と突合し、
   「追い風が重なる/矛盾する」業種を各2-3個。
4. **当日イベント**: 決算・指標・SQ 等の時刻付きリスト。

### Step 4 記録と出力
- `output/jp_open/predictions.jsonl` に append（インライン python で。フォーマット固定）:
  `{"date","made_at","implied_gap_pct","fut_source","predicted_open","predicted_day",`
  `"confidence","drivers":[...],"realized":null}`
  （predicted_* は up/flat/down。flat は ±0.3% 目安）
- `output/jp_open/brief_{YYYYMMDD}.md` に全文保存。
- ユーザーへ1画面: 定型注記 → 寄付 → シナリオ → セクター → イベント → 的中率サマリ。

## 運用メモ
- 最適な実行時刻は 7:00-8:50 JST（CME 終値確定後・寄付前）。/us-brief → /jp-open の
  チェーンを /schedule で毎朝自動化してよい（人間が設定）。
- 的中率が上がらないことは失敗ではない（方向は予測不能という中心知見と整合）。
  むしろ confidence=medium の的中率が low を下回り続けるなら過信の証拠＝要報告。
