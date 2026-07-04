# research_ops — 運用系スキルの状態ディレクトリ

research_loop/（研究サイクル）と対をなす**運用側**の永続状態。各スキルの決定論的コアが
書き込み、スキル（LLM）は解釈・提案・レポートのみを行う。

| ファイル/ディレクトリ | 書き手 | 内容 |
|---|---|---|
| `data_qa_log.jsonl` | `/data-qa`（examples/data_qa.py） | 監査 findings の append-only ログ |
| `data_contract.json` | 〃（--init-contract 時のみ） | 主要データディレクトリのスキーマ契約 |
| `data_qa_reports/` | `/data-qa` | 監査レポート（解釈・修復提案・実施記録） |
| `monitor_config.json` | 人間（認定時に凍結） | 稼働スリーブの認定値（m・tol・σ_max 等）＝監視の帰無仮説 |
| `monitor_log.jsonl` | `/strategy-monitor`（examples/strategy_monitor.py） | 月次判定の append-only ログ（PSR・e値・CUSUM） |
| `monitor_reports/` | `/strategy-monitor` | 月次健全性レポート |
| `acquisitions/` | `/data-acquire` | データ調達の記録（出所・取得日・検証結果・再現スクリプトへの参照） |
| `data_update_log.jsonl` | `/data-update` | 一括更新の実行記録（ソース別成否・追加日数・QA検収結果） |
| `data_ideas.md` | `/data-scout` | オルタナ/新規データソース台帳（D-n・取得待ち行列＝/data-acquire の入力） |
| `surveys/` | `/data-scout` | データソース調査レポート |
| `PENDING.md` | 全スキル→人間 | 承認/判断の待ち行列（open に依存するアクションは実行禁止） |
| `loop_lint_baseline.json` | 人間（--freeze-a・上限変更） | §A ハッシュ・K 予算（週次/scope）＝機械強制の基準 |
| `operator.lock` / `operator_log.jsonl` / `briefs/` | `/operator` | 単一ライターロック・判断ログ・日次ブリーフ |
| `retro/` | `/loop-retro` | ループ振り返りレポート |
| `redteam/` | `/red-team` | 事前登録の敵対的レビュー記録 |
| `replications/` | `/replicate` | 論文複製プロトコルと結果 |

ハッシュ台帳等の機械ローカル状態は `data/qa/`（gitignore 圏）。

**原則**: ①append-only の時刻印ログ＋期待値との定期 diff（全スキル共通の骨格）
②数値はコードが計算・LLM は翻訳と提案のみ ③市場データの自動修復禁止（flag, don't clean）
④キルスイッチ等の重大操作は常に人間ゲート。
