# CLAUDE.md — strategy_factory 作業規約

日本株クオンツの**検証ファクトリ**（仮説を厳密に裁く装置＝成果物）＋ Phase 2 無人運用。
現状と中心知見の正本は `docs/03-research-findings.md`（人間キュレーション。要旨: 独立エッジは
ほぼ全滅、value のみ控えめに耐久。稼働は value↔PEAD switch ＋ TSMOM の2スリーブ）。

## 環境・コマンド（Windows 主体）

- Python 3.13 / venv: `.\.venv\Scripts\python.exe`（Linux/macOS は `.venv/bin/python`）
- テスト: `.\.venv\Scripts\python.exe -m pytest -q`
- スクリプト実行（日本語出力の化け対策を必ず付ける）:
  `$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\<x>.py`
- データ更新は `/data-update` スキル経由（ToSTNeT は約2週間で消えるため最優先）

## ディレクトリ

- `invest_system/` — ライブラリ本体（`validation/` の DSR・CPCV・試行レジストリが背骨）
- `examples/` — 1 scope = 1 研究スクリプト（`research_<scope>.py`。雛形は PLAYBOOK Step 4 参照）
- `docs/` — 番号付き。01=ナレッジ, 02=設計, **03=知見正本**, **24=データカタログ**, 45以降=事前登録
- `research_loop/` — BACKLOG / IDEAS / **PLAYBOOK.md（サイクル手順書）** / HEURISTICS / LESSONS / ledger.jsonl
- `research_ops/` — RUNBOOK.md（無人運用）/ PENDING.md（人間宛て保留）/ operator ログ類
- `data/` — gitignore 済みキャッシュ（**J-Quants 規約によりコミット禁止**）。レジストリ = `data/research_trials.db`

## 研究規律（最重要 — 違反は成果より重い）

- **事前登録なしにバックテストを回さない**。手順は `research_loop/PLAYBOOK.md`。§A は不変（人間のみ編集可）
- 合否は **DSR≥0.95 のみ**。判定済み scope の再検証禁止（BACKLOG §2 除外リスト・レジストリと必ず dedupe）
- **PIT 必須**: `AsOfView` / `point_in_time_universe` / `execution_lag=1`（または T+1 始値）/
  fundamentals `lag_days≥1` / 上場廃止込みリターン。**OOS（2024-01以降）を設計・選択に使わない**
- **K 会計**: 走らせず棄却した候補も `extra_trials` で計上。レジストリは registry API 経由のみ（直接 SQL 禁止）
- kaiten 等**レジストリ外バックテストは in-sample**＝設計根拠・採用判定に使わない（judge 系＋真の OOS が必須）
- PASS しても採用は人間ゲート（`research_loop/proposals/` に提案を書いて停止）
- データ品質は flag, don't clean（自動修復しない。`/data-qa`）

## 接触禁止（自律セッション）

`examples/phase2_*.py`・`invest_system/production/**`・ops リポジトリへの書き込み・
`docs/03`・`data/research_trials.db` への直接 SQL・PLAYBOOK §A。

## スキル（研究ループは必ずこれ経由で回す）

- `/operator` — 無人運用の唯一のエントリポイント（`plan` 引数で判断のみ）
- 研究: `/research-cycle` `/research-scout` `/red-team` `/replicate` `/loop-retro`
- データ: `/data-update` `/data-qa` `/data-acquire` `/data-scout`
- 運用監視: `/phase2-triage`(夜次ランの日次トリアージ・診断のみ) `/strategy-monitor`（月次健全性）
  `/ops-review`（月次レビュー・Phase 2 を操作しない）
- 裁量・運用補助（**裁量レーン**＝出力は `output/` のみ・研究/レジストリへの入力禁止。
  原則: 数値=ローカル計算、鮮度/文脈=WebSearch で毎回補完＝ローカルに閉じない）:
  `/market-brief` `/us-brief` `/jp-open`(朝チェーン: us→jp) `/news-scan`
  `/kaigai-scan`(英語圏の日本株談義・海外勢フロー突合)
  `/kabu-dd` `/kabu-screen` `/kabu-funda`(altデータ+SNS深掘り) `/kessan-preview`
  `/tactical-brief` `/pretrade-check` `/sairyo-retro`(裁量レーンの月次答え合わせ・淘汰)
- 規律補助: `/registry-ask`(制度記憶の照会・読み取り専用) `/backtest-audit`(レジストリ外BTの監査)

## Phase 2（実運用・ペーパー先行）

- 毎晩 21:30 JST に GitHub Actions（private repo `futnag/strategy-factory-ops`、
  ローカル `C:\Users\futos\strategy-factory-ops`）→ Supabase → Vercel ダッシュボード。失敗時は Issue 起票
- 接続情報は `.env` と ops repo の Secrets。**キー・パスワードをコード/git/チャットに書かない**

## Git

- コミット規約: `research(<scope>): 要約` / `ops(<領域>): 要約` / `data(<領域>): 要約`（日本語可）
- **push はユーザーの明示依頼時のみ**。研究サイクルの自動コミットは PLAYBOOK §A-10 の2回まで
- 市場データ（`data/`）・`.env` は絶対にコミットしない
