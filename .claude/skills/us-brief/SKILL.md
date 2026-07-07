---
name: us-brief
description: 米国市場の状況分析（前夜の米指数・金利・VIX・FX/商品＋日本への橋渡し japan_handoff）。数値は examples/us_brief.py が計算、鮮度は update_external.py で先に確保、定性（なぜ動いたか・時間外決算・今夜の指標）は WebSearch で補完。/jp-open の上流。裁量レーン＝研究・事前登録の入力にしない。引数: --asof YYYY-MM-DD（省略時=今日）。
---

# /us-brief — 米国市況ブリーフ（1起動＝1ブリーフ）

決定論的コアは `examples/us_brief.py`。出力 JSON（`output/us_brief/brief_{YYYYMMDD}.json`）の
`japan_handoff` 節（NK225先物リターン→寄付ギャップ推計・USDJPY）は **/jp-open の入力**になる。

## ガードレール（違反不可）

1. **数値はコードが正**。LLM は解釈・定性補完のみ（指数値や変化率を Web 記事の数字で
   上書きしない。Web は「理由」と「速報値の目安」担当）。
2. 実行してよい書き込み系は `examples/update_external.py`（検証付き追記・冪等）のみ。
   それ以外の書き込みは `output/` に限る。
3. **裁量レーン**: research_loop/・事前登録・レジストリの設計根拠に使わない。
4. スクリプトの [WARN]（系列停滞）が残ったまま解釈する場合は、レポート冒頭に
   「◯◯は X日前の値」と必ず明記（FRED系の金利・VIXは数日遅れが仕様）。

## 手順

### Step 1 鮮度確保
```powershell
$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe examples\update_external.py
```
失敗・部分成功でも続行してよい（Step 2 の WARN が状態を教えてくれる）。

### Step 2 計算
```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\us_brief.py
```

### Step 3 Web で定性補完（WebSearch。各項目に出典を付す）
1. 前夜の米市場サマリ（なぜ上げ/下げたか。Reuters・Bloomberg・CNBC 級を優先）
2. 主導セクター・大型テック/半導体（SOX はローカルに無いので Web で水準と騰落を取る）
3. 引け後決算・ガイダンス（日本の関連銘柄に効くもの優先: 半導体・ハイテク・自動車）
4. 今夜〜今週の米イベント（FOMC・CPI・雇用統計・要人発言・国債入札）
5. 日経先物の現値（CME/大取ナイト。JSON の implied_open_gap_pct は日足近似なので
   寄付前はこちらが正）

### Step 4 解釈レポート（1画面）
- **3行サマリ**: リスク選好・金利・ドル円の方向感。
- **変化点**: 前回 brief_*.json との差分のみ（毎回全部を書かない）。
- **日本への含意**: japan_handoff の数値＋セクター対応（NASDAQ/SOX→電機・半導体、
  米金利→銀行/グロース、WTI→石油・商社、USDJPY→輸出/内需）。詳細な当日見通しは
  /jp-open の仕事＝ここでは1-2行に留める。
- **今夜の注視点**。

## 運用メモ
- 主用途は朝（JST）実行→そのまま /jp-open へチェーン。夕方実行なら「今夜の米市場の
  注目点」モードとして Step 3-4 の重心を予定側に移す。
- WTI 等の商品は継続足のロール起因の段差があり得る（1ヶ月変化が異様に大きい時は
  Web でクロスチェックしてから解釈する）。
