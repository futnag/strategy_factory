---
name: pretrade-check
description: リバランス発注前（月末注文生成の翌朝）の最終チェックリスト。data/phase2 の orders/intended/manifest/status を読み取り専用で検査し、資本整合・ロット・集中度・ADV比・値幅/貸借・キルスイッチ距離・決算またぎを一覧化して GO/CAUTION/HOLD を提案。Phase 2 への書き込み・発注は一切しない（人間ゲート維持）。
---

# /pretrade-check — 発注前チェック（1起動＝当月注文の検査1枚）

/ops-review（月次・事後）の対になる**事前**チェック。判断材料を揃えるのが仕事で、
判断と発注は人間。

## ガードレール（違反不可）

1. **読み取り専用**: `data/phase2/`・ops リポジトリ・`invest_system/production/` に書かない。
   実行してよい既存スクリプトは `examples/phase2_reconcile.py` のみ（冪等・読み取り系）。
2. **発注しない・注文を修正しない**: 問題を見つけたら HOLD 提案と修正案の説明まで。
   注文の再生成が必要なら「人間が phase2_generate_orders.py を再実行」と案内する。
3. 書き込みは `output/phase2_pretrade/` のみ。
4. 検査で使う閾値は docs/02 D5・manifest の値が正（自分で発明しない。不明なら UNKNOWN と報告）。

## 手順

### Step 1 対象月の成果物を読む
`data/phase2/` から当月（無ければ最新）の
`orders_eq_*.csv` / `orders_ts_*.csv` / `intended_*.parquet` / `manifest_{YYYY-MM}.json` /
`status.json`（＋必要なら `report_latest.md`）。
manifest の資本・判定日・ヘッジ仕様を控える。鮮度が古い（当月分が無い）場合は
「今月はまだ注文生成前」と報告して終了。

### Step 2 検査（各項目 PASS/CAUTION/FAIL/UNKNOWN）
1. **整合**: 注文合計額 ≒ manifest 資本×スリーブ配分。intended と orders の突合
   （差分銘柄ゼロか）。可能なら `phase2_reconcile.py` を実行し前提状態を最新化。
2. **ロット/単元**: 単元未満株の扱いが D5 仕様どおりか、端数・最小約定金額。
3. **集中度**: 1銘柄の対資本比が異常でないか（intended の最大ウェイト）。
4. **流動性**: 各注文の金額 ÷ 直近20日 ADV（売買代金, ローカル daily から計算）。
   participation>10% は CAUTION、>30% は FAIL。
5. **執行リスク**: 直近でストップ高安に張り付いた銘柄（UL/LL フラグ）、貸借区分、
   直近の TDnet 重要開示・ToSTNeT 超大口の該当。
6. **決算またぎ**: 各銘柄の次回決算予定（`equities/events.py` の
   `days_to_next_announcement` か EDINET DB カレンダー）。発注〜数日内に決算がある銘柄を列挙
   （警告であり除外指図ではない＝除外オーバーレイは検証済み FAIL、docs/03）。
7. **キルスイッチ距離**: status.json の DD・閾値までの距離（strategy-monitor の領分だが
   発注日の再確認として表示）。
8. **当日朝の材料（Web・必須）**: 発注銘柄の当日朝の新着材料（決算・適時開示・ニュース・
   ADR/PTS の急変）を WebSearch で確認。ローカル TDnet ミラーは前日分まで＝当日朝は Web が正。

### Step 3 出力
`output/phase2_pretrade/check_{YYYYMMDD}.md`:
検査表（銘柄×項目）＋総合判定 **GO / CAUTION（列挙付き）/ HOLD（理由と人間の次アクション）**。
ユーザーには総合判定と CAUTION/FAIL 項目だけを1画面で。

## 運用メモ
- 実行タイミング: 月末注文生成の翌朝（寄付前）。それ以外の日に呼ばれたら
  「直近注文の事後点検モード」として同じ検査を行い、その旨を明記。
- 実弾移行後は fills_actual の存在も確認対象に追加（置き忘れ検知）。
