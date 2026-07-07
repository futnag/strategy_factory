# 64 - 裁量・運用補助スキル群（設計と運用）

**version 1.0（2026-07-07）**。本ドキュメントは、研究ループ用スキル（/operator・/research-cycle 等）
の外側に整備した**裁量レーン・運用補助・規律補助のスキル群15本**の設計正本。
個々の手順の正本は各 `.claude/skills/<name>/SKILL.md`（矛盾時は SKILL.md が優先）。

- 経緯: 2026-07-07 のセッションでユーザー要望に基づき一括整備（過去のチャット履歴・
  作業パターンの分析から提案→承認）。
- 位置づけ: **研究レーン（レジストリ・DSR・事前登録）とは絶縁した別レーン**。
  ここでの分析・予想・提案は投資判断の補助であり、検証済みエッジではない。

---

## 1. レーン構造と絶縁原則

```
研究レーン   /research-cycle 等 … PIT・事前登録・K会計・DSR≥0.95（docs/03 が知見正本）
運用レーン   Phase 2（無人・ルール駆動）… 人間ゲートでのみ変更
裁量レーン   本スキル群 … 出力は output/ のみ・上2レーンへの入力/操作を禁止
```

絶縁の根拠:
1. **PIT 不能**: Web・SNS・ニュース由来の情報はバックフィル不可＝検証不能。研究の設計根拠に
   使えば事前登録規律が崩れる（PLAYBOOK §A-6）。
2. **in-sample 教訓**: レジストリ外の分析は多重検定・後知恵の統制がない（kaiten の教訓）。
   裁量観察を研究に載せる正規ルートは /research-scout → BACKLOG → 事前登録のみ。
3. **人間ゲート**: 発注・Phase 2 変更・ルール変更は常に人間（docs/02 D5）。スキルは提案止まり。

## 2. 設計原則（全スキル共通）

| # | 原則 | 内容 |
|---|---|---|
| P1 | 数値はコード | 定量は `examples/*.py` かインライン python が計算。LLM は解釈・比較・定性のみ（数値の創作・上書き禁止） |
| P2 | Web 鮮度 | ローカルミラーは数時間〜数日遅延＝**鮮度と文脈は WebSearch で毎回補完**（必須ステップ）。ただし Web の数字でローカル計算値を上書きしない。「数値は X 日時点・文脈は本日」と時点を分けて書く |
| P3 | 出力先 | `output/<skill>/` のみ（gitignore 圏内）。research_loop/・docs/・data/phase2/・ops repo に書かない |
| P4 | エビデンス格付け | [一次]（開示・IR）/[機関]（FM調査・規制文書）/[報道]/[UGC]/[未確認] のラベル＋出典＋日付。UGC 単独は根拠にしない（操作可能） |
| P5 | 記録なき予想の禁止 | 予想・提案は必ずファイルに記録（jp-open は predictions.jsonl 必須）。月次 /sairyo-retro が答え合わせと淘汰を行う |
| P6 | 断定禁止 | 投資助言ではない。条件付きケース（「◯◯なら△△寄り」）で書く。confidence に high を使わない |
| P7 | ToS 遵守 | 単発の WebFetch/WebSearch のみ。自動巡回・大量取得・ログイン壁の回避をしない |

## 3. スキル一覧（15本）

### 市場観測
| スキル | 役割 | 数値コア | 出力 | 推奨頻度 |
|---|---|---|---|---|
| /market-brief | 日本市場の定点観測（指数・セクターRS・部門別フロー・信用/空売り・IV・マクロ） | examples/market_brief.py | output/market_brief/ | 週次 |
| /us-brief | 米国市況＋日本への橋渡し（japan_handoff: NK先物→寄付ギャップ推計） | examples/us_brief.py | output/us_brief/ | 毎朝 |
| /jp-open | 当日の日本株見通し（寄付=機械的推計と日中=シナリオを分離、予想ログで自己校正） | us_brief JSON | output/jp_open/ | 毎朝 |
| /news-scan | 市場に効くニュース→スリーブ/watchlist への影響マップ | — | output/news/ | 随時 |
| /kaigai-scan | 英語圏の日本株談義（3層収集・日英ギャップ・機関ナラティブ・実フロー突合） | flows（ローカル） | output/kaigai/ | 月1-2 |

### 個別株（動線: screen → dd → funda → kessan-preview）
| スキル | 役割 | 出力 | 備考 |
|---|---|---|---|
| /kabu-screen | スクリーニング（EDINET DB＋ローカルファクター、Web 検品必須） | output/kabu/ | 10x プリセットは docs のマイクロキャップ・ブリーフィング参照 |
| /kabu-dd | 定型デューデリ（財務/株主/需給/バリュエーション分位＋Web 最新確認） | output/kabu/ | --peers/--forensic/--watch |
| /kabu-funda | 深掘りファンダ（altデータ＋SNS。顧客と投資家のセンチメント分離、監視KPIリスト） | output/kabu/ | 有報叙述変化の検知に disclosure_text.parquet を転用 |
| /kessan-preview | 決算プレビュー（会社予想・進捗・過去反応＋Web コンセンサス）。発表後はレビューモード | output/kabu/ | PEAD スリーブと絶縁 |

### 投資行動・運用補助
| スキル | 役割 | 出力 | 備考 |
|---|---|---|---|
| /tactical-brief | 行動提案（レーンA=ルール内・レーンB=裁量サテライト。3点セット必須） | output/tactical/ | レジストリの負知見と矛盾する案を自動棄却 |
| /pretrade-check | 発注前チェック（資本整合・ロット・ADV比・値幅/貸借・決算またぎ・当日朝の材料） | output/phase2_pretrade/ | data/phase2 読み取り専用・GO/CAUTION/HOLD |
| /phase2-triage | 夜次ランの日次トリアージ（既知障害パターン表・安全な復旧のみ実行） | output/phase2_triage/ | 状態変更（rerun/push/Issue）は人間確認後 |

### 規律補助
| スキル | 役割 | 出力 | 備考 |
|---|---|---|---|
| /registry-ask | 制度記憶の照会（既試/類似/未踏を scope・K・DSR・失敗型付きで即答） | （読み取り専用） | DB 直 SQL 禁止・registry API/loop_status 経由 |
| /backtest-audit | レジストリ外バックテストの監査（8項目チェックリスト＋judge_grid 移植プラン） | output/audit/ | 初戦想定は kaiten 系 |
| /sairyo-retro | 裁量レーンの月次答え合わせ（校正・レーンB台帳・KPI追跡・90日未使用の淘汰） | output/sairyo/ | 本レーン全体のメタ監査 |

## 4. 推奨運用サイクル

- **毎朝**: /phase2-triage → /us-brief → /jp-open（この順。update_external の重複を避ける）
- **週次**: /market-brief（フロー・信用は週次更新）→ 必要に応じ /news-scan /tactical-brief
- **月次**: 月末リバランス翌朝 /pretrade-check。月中に /sairyo-retro（/ops-review・
  /strategy-monitor と同時期）。/kaigai-scan は FM調査公表・BOJ 会合後が好機
- **随時**: 銘柄動線（screen→dd→funda→kessan-preview）、/registry-ask、/backtest-audit

## 5. 計算スクリプトと出力データ

- `examples/market_brief.py` / `examples/us_brief.py`: 読み取り専用・ネットワーク不要・
  欠損は WARN で劣化継続・常に exit 0。JSON を日付付きで `output/` に蓄積（前回比較の材料）。
  業種指数名は上場マスタ S33 順から導出（ハードコード最小化）。
- `output/jp_open/predictions.jsonl`: 予想ログ（append-only。/jp-open が記入・答え合わせ、
  /sairyo-retro が校正を集計）。
- `output/` は gitignore 圏内＝コミットされない（成果物はローカル蓄積）。

## 6. 既知の限界

- ローカルミラーの遅延（指数=週次 refresh 等）→ P2 の Web 補完と鮮度ゲートで運用対処。
- 米金利・VIX（FRED系）は数日遅れが仕様。当日値は Web で。
- ペイウォール（FT/Bloomberg）は見出しまで。X は API なし＝site: 検索経由。
- jp-open の的中率は偶然域に留まる想定（方向予測不能は docs/03 の確定知見）。目的は校正の
  記録であり、的中ではない。

## 7. 変更履歴

| 日付 | 変更 | 根拠 |
|---|---|---|
| 2026-07-07 | v1.0 全15本を整備・本ドキュメント作成 | ユーザー承認済み提案（セッション記録） |
| 2026-07-07 | P2（Web 鮮度原則）を全スキルに適用 | ユーザーフィードバック「ローカルに閉じず常に最新を広く」 |
