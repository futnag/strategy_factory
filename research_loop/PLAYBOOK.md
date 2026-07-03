# PLAYBOOK — 自律研究サイクル手順書

**version 1.0（2026-07-03）**。`/research-cycle` で起動されたセッションが従う手順書。

- **§A 不変ガードレール**は**人間のみ編集可**。ループ（自律セッション）は §A を一切変更しない。
  §A は `.claude/skills/research-cycle/SKILL.md` にも複製されており、矛盾時は SKILL.md が優先。
- **§B 手順**はループが改訂できる（1サイクル最大1件・§C 改訂履歴に append-only で記録）。
- 目的: 検証ファクトリ（`invest_system/` ＋ 永続レジストリ）の規律の内側で、
  「仮説考案 → Stage-0 → 事前登録 → 実装 → 実行 → DSR判定 → 知見記録 → 自己改善」を1コマンドで回す。

---

## §A 不変ガードレール（IMMUTABLE — 人間のみ編集可）

1. **1サイクル＝1仮説＝1 judge_grid scope**。グリッドは **≤6セル**、サイクルの K 予算は
   **≤8（`extra_trials`・`log_scan_trials` 込み）**。超える場合は事前登録 doc に理由を明記して
   事前に正当化する（事後の拡張は不可）。
2. **合否判定は DSR≥0.95 のみ**（DP18）。robustness・PBO・minTRL・サブ期間・容量は表示/診断専用。
   判定基準の緩和・変更・別基準の発明は禁止。事前登録した**副次基準**（買い持ち超過等）は
   「PASS の追加条件」としてのみ使える（緩和方向には使えない）。
3. **事後救済の禁止**: 結果を見た後のシグナル定義・グリッド・ユニバース・期間の変更は
   **新仮説＝BACKLOG 行き（次サイクル）**。同一サイクル内で続行できるのは、事前登録 doc §3 に
   **事前に段階として書かれた stage のみ**（docs/45 の Stage1→1.5 型）。バグ修正は可・シグナル変更は不可。
4. **接触禁止**: `examples/phase2_*.py`・`invest_system/production/**`・ops リポジトリ（読み取りのみ可）・
   `docs/03-research-findings.md`（人間キュレーション正本）・`data/research_trials.db` への直接 SQL
   （registry API 経由のみ）・本 §A 自身。
5. **PASS しても採用しない**: `research_loop/proposals/` に採用提案 doc を書いて**停止**。
   実運用（Phase 2）への反映は人間ゲート。
6. **PIT チェックリスト**（実装前に全項目を確認し、事前登録 doc に確認済みと記載）:
   - シグナルは ≤t 情報のみ（`AsOfView` 経由）
   - 執行は `execution_lag=1`（T+1）または `open_fill_backtest`（T+1始値）
   - fundamentals は `lag_days=1` 以上（`point_in_time` / `fundamentals_panel`）
   - ユニバースは `point_in_time_universe`（生存者バイアス除去）
   - リターンは `forward_returns_with_delisting`（上場廃止込み）
   - **OOS ホールドアウト（2024-01 以降）を設計・選択に使わない**（判定時の診断表示のみ）
   - kaiten 系の結果はレジストリ外（in-sample）＝設計根拠に使わない
7. **K 会計**: グリッド各セル＝1試行（`judge_grid` が自動記録）。走らせる前に棄却した探索候補も
   `extra_trials` / `log_scan_trials` で必ず K に算入。パラメータ比較を伴わない記述スキャン
   （分布・件数の確認のみ）は K 外。
8. **コスト床**（これ未満の想定は事前登録 doc で正当化必須）: 月次クロスセクション 15bps／
   日次イベント系 30bps／ショート貸株 ≥115bps（イベント系・小型は 300bps）／
   ストップ高安は `limit_lock_flags` 適用／容量は `adv`×participation=0.1 で算定。
9. **同時に2サイクルを走らせない**（ledger.jsonl・BACKLOG.md は単一ライター前提。
   スケジュール化する場合もこの直列制約を維持）。
10. **git 自動コミットはサイクル毎2回まで**（①実行前の事前登録 doc、②結果確定後の成果物）。
    コミット対象は `docs/`・`examples/`・`research_loop/`・`tests/` のみ。push はしない（人間が行う）。

---

## §B 手順（サイクルの10ステップ — ループが改訂可）

### Step 0 状態ブートストラップ
```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\loop_status.py
```
- `research_loop/HEURISTICS.md`・`LESSONS.md`（タクソノミー＋直近10件）・`BACKLOG.md`・
  `ledger.jsonl` 末尾5行を読む。
- `git status` を確認: **追跡中ファイルに未コミット変更があれば**人間に報告して停止
  （無関係な未追跡ファイルは無視してよい）。
- cycle_id = `YYYY-MM-DD-<slug>` を決める。

### Step 1 仮説の選択または生成
- BACKLOG の最優先 ⬜ を1つ取る（🔬 に更新）。
- ⬜ が無い/低EVなら、4領域（a:日本株XS / b:イベント駆動 / c:時系列・マルチアセット / d:クリプト）に
  **散らして3〜5候補**を生成する。生成時は HEURISTICS の全項目を prior として適用。
- **dedupe**: registry の scope 一覧（loop_status 出力）・BACKLOG §2 除外リスト・§3 Stage-0 棄却台帳・
  LESSONS と突合。既試の再発明は禁止（「同じ経済機構＋同じデータ軸」なら別名でも既試とみなす）。
- 新候補は TODO.md の項目構造（**仮説／経済的根拠／データ／新規性／独立性／注意**）で BACKLOG §1 に追記。

### Step 2 Stage-0 キルチェック（バックテスト無し・K=0）
選んだ候補に以下を順に適用。1つでも fail → BACKLOG §3 棄却台帳に理由付きで記録し、
次候補へ（全滅なら verdict="no_viable" として Step 8 へ）:
1. **データ実在性**: `docs/24-data-catalog.md` と突合。必要な列・期間・PIT アンカーが実在するか。
2. **minTRL/MinBTL 実現可能性**: 想定 SR・n_obs・K で DSR≥0.95 が構造的に到達可能か
   （`invest_system/validation/dsr.py` の `min_backtest_length`/`min_track_record_length` で概算。
   月次×単一系列×低SR は即死）。
3. **直交性 prior**: momentum(12-1)・value(B/M) の単調変換になっていないか（見込み ρ̄>0.3 → kill か残差化を設計に組み込む）。
4. **コスト損益分岐**: 想定回転×現実スプレッド < 想定 gross か（日次・小型は片道25bpsが実測死線）。
5. **容量下限**: 想定容量が ¥数千万未満なら kill。
6. **de-risk 形状**: タイミング/ゲート系なら「買い持ちベンチ超過」を合格の副次基準に入れられるか。
- 必要なら**記述スキャン**（イベント件数・分布の確認のみ・パラメータ比較なし＝K外）を最初に行ってよい。

### Step 3 事前登録
- `docs/NN-<scope>-preregistration.md` を書く（NN＝docs/ 内の最大番号+1）。docs/48 の型に従う:
  §0 既知の失敗型を構造的に回避する設計表 / §1 実現可能性（記述スキャン結果）/
  §2 事前登録（仮説 H1..Hn・経済的根拠・データ＆PIT 確認・**固定グリッド表と K**・judge 配線
  （scope/costs/frictions/universe）・**合格基準＝DSR≥0.95＋事前固定の副次基準**・既知の限界）/
  §3 実装ステージ（同一サイクル内で許される段階を事前列挙）/ §4 出典 / §5 結果（空欄）。
- **コミット①**: `git add docs/NN-*.md research_loop/` →
  `research(<scope>): preregistration (docs/NN)`。**実行前にコミットすること**（改竄防止のタイムスタンプ）。

### Step 4 実装
- `examples/research_<scope>.py` を書く。雛形: `research_asset_growth.py`（XS）／
  `research_limit_reversal.py`（イベント）／`research_tsmom_multiasset.py`（時系列）。
- 再利用: `equities/{panel,universe,fundamentals,factors,frictions}` ＋ `research/data_view.AsOfView` ＋
  `research/strategy.py` テンプレート ＋ `judge_grid(..., registry=default_registry())` ＋ `write_html`。
- IS/OOS（2024-01 境界）分割診断と独立性診断（ρ̄ vs value/momentum/size）を必ず含める。

### Step 5 実行
```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\research_<scope>.py
```
- バグ修正の反復は可。**シグナル定義・グリッドの変更は不可**（§A-3）。

### Step 6 判定
- verdict は `GridVerdict.passed`（＋事前登録済み副次基準）のみで決める。
- 事前登録 doc の §5 に記入: verdict / 主要統計（SR・PSR・DSR・minTRL・maxDD・容量）/
  機構診断（gross/net・コスト感度・サブ期間・脚分解・独立性）/ **失敗型分類**
  （LESSONS §1 の既存型に照合。新型を立てる場合は既存型で説明できない理由を明記）。

### Step 7 記録
- `LESSONS.md` §4 に1エントリ追記（scope・verdict・DSR・失敗型・機構1行・成果物リンク）。
- `BACKLOG.md` の当該項目を ❌/✅ に更新。派生アイデアがあれば新規 ⬜ として追記。
- `ledger.jsonl` に1行追記（スキーマは README 参照）。

### Step 8 振り返り＝自己改善
- 「**どの HEURISTICS があれば Stage-0 でこの結果を予見できたか**」を自問し、
  H-n を追加/強化（本サイクルの scope・doc を証拠として引用。証拠なしの追加は禁止）。
- 新しい失敗型が出た場合のみ LESSONS §1 タクソノミーに追記。
- §B の手順自体に改善があれば**最大1件**改訂し、§C に記録。§A には触れない。
- no_viable サイクルでも必ずここを実施（生成の質を上げる）。

### Step 9 PASS 時のみ: 採用提案
- `research_loop/proposals/YYYY-MM-DD-<scope>.md` を書く:
  戦略仕様（再現手順）／判定統計／容量・回転・執行前提／既存2スリーブ（value+PEAD LT・TSMOM）との
  リターン相関と結合効果試算／推奨サイジング／リスクと監視指標／Phase 2 統合手順**案**（実施は人間）。
- 提案を書いたら**そこで停止**（Phase 2 側のファイルには触れない＝§A-4,5）。

### Step 10 コミット②とサマリ
- `git add examples/research_<scope>.py docs/NN-*.md research_loop/ tests/`（新テストがある場合）→
  `research(<scope>): <verdict> — <1行要約> (docs/NN §5)`。
- ユーザー向けに1画面のサイクルサマリを出力:
  仮説／verdict／DSR／失敗型 or 提案／K 消費／自己改善（追加した H-n・改訂）／次サイクル候補。
- 運用ノート: 月1回程度、`data/research_trials.db` を `data/backup/` へコピーする
  （gitignore 圏内の手元バックアップ。git には含めない）。

---

## §C §B 改訂履歴（append-only）

| 日付 | cycle_id | 変更 | 根拠 | 証拠 |
|---|---|---|---|---|
| 2026-07-03 | (初版) | §B v1.0 制定 | docs/03・docs/45-48・TODO.md の運用実績を手順化 | — |
