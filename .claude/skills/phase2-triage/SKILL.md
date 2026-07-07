---
name: phase2-triage
description: Phase 2 夜次パイプライン（毎晩21:30 JST・GitHub Actions futnag/strategy-factory-ops → Supabase → Vercel）の日次トリアージ。昨晩のランを gh で確認→失敗なら既知障害パターンと照合→ローカルで安全な復旧（update_external/reconcile の再実行）→成果物整合と鮮度の検証→1枚レポート。診断と提案が主・ops repo と Phase 2 状態への書き込みは一切しない。ops-review（月次）と strategy-monitor（月次）の間を埋める日次の穴埋め。
---

# /phase2-triage — 夜次ランの日次トリアージ（1起動＝昨晩1ラン分）

「昨晩のランは無事だったか。ダメならどこが・なぜ・今どうするか」に5分で答える。
成功日は3行で終わる（静穏時に長文を書かない）。

## ガードレール（違反不可）

1. **書き込み禁止**: ops リポジトリ（ローカルクローン含む）・`data/phase2/`・
   `invest_system/production/**` に書かない。実行してよいのは冪等な既存スクリプト
   `examples/update_external.py`・`examples/phase2_reconcile.py` のみ。
2. **状態変更系は人間確認後**: `gh run rerun`・Issue の作成/クローズ・
   `phase2_dashboard_data.py --out`（ops repo への書き出し）・Supabase push は、
   提案として明示し人間の指示を待つ（Actions が Issue を起票する体系を壊さない）。
3. **秘密情報を出力しない**: ログ引用時に Secrets・接続文字列・キーが混ざったらマスク。
   `.env` の中身を表示しない。
4. 書き込みは `output/phase2_triage/` のみ。原因が分からない場合は「不明」と報告する
   （推測で復旧コマンドを打たない）。

## 手順

### Step 1 昨晩のラン確認
```powershell
gh run list -R futnag/strategy-factory-ops --limit 5 --json databaseId,displayTitle,conclusion,createdAt,url
gh issue list -R futnag/strategy-factory-ops --state open
```
最新ランが success で open Issue も無ければ → Step 3 の鮮度確認だけして3行報告で終了。
失敗なら `gh run view <id> -R futnag/strategy-factory-ops --log-failed` で失敗ステップと
エラー本文を特定する。

### Step 2 既知障害パターン照合（内蔵パターン表）
| パターン | 兆候 | 一次対応 |
|---|---|---|
| 外部価格ソース障害（Yahoo 等） | update_external 段で nk225_fut/usdjpy 欠損・HTTP エラー | ローカルで update_external 再実行（検証付き追記・冪等）。ソース側の障害か Web で確認 |
| J-Quants レート/遅延 | 429・タイムアウト・by-date 欠損 | 翌晩自然回復が既定。急ぎなら /data-update を人間に提案 |
| Supabase 接続失敗 | pooler 接続エラー・認証失敗 | パスワード/ロール/一時障害の切り分けを提案（push は人間） |
| Vercel デプロイ失敗 | deploy 段の失敗 | Vercel MCP の get_deployment_build_logs でビルドログ確認 |
| Actions cache 消失 | J-Quants ミラーの再シードから開始・実行時間激増 | 仕様（再開可能）。完走していれば対応不要、失敗なら再ラン提案 |
| 月末のみ: 注文生成系 | generate_orders 段の失敗 | **月末は要即応**＝人間へ最優先で報告（/pretrade-check の前提が崩れる） |
新パターンだったら表への追記**案**をレポートに含める（このファイルの編集は人間承認後）。

### Step 3 成果物の整合と鮮度（成功/失敗どちらでも）
- ローカル: `data/phase2/status.json` の日付・`equity_daily.csv` 末尾日が直近営業日か。
  必要なら `phase2_reconcile.py` を実行して照合を最新化（冪等・実行可）。
- クラウド: ダッシュボード（URL は ops repo 参照）の最終更新表示、または Supabase の
  最新行日付（MCP execute_sql が使えるとき・SELECT のみ）。
- ローカルとクラウドの日付がズレていたら同期漏れとして原因ステップを特定。

### Step 4 レポート
`output/phase2_triage/triage_{YYYYMMDD}.md`:
①ラン結果（成功/失敗・所要・URL） ②原因分類（パターン名 or 不明＋証拠ログ抜粋）
③実施した安全な復旧（update_external/reconcile の結果） ④人間への提案アクション
（rerun・push・Issue 対応の要否） ⑤鮮度表（ローカル/クラウド各成果物の最終日付）。
ユーザーへは1画面（成功日は3行）。

## 運用メモ
- 実行タイミング: 朝（/us-brief より前が理想＝update_external の重複実行を避けられる）。
  /schedule で毎朝自動化してよい（人間が設定）。
- 3日連続で同一パターンの失敗 → 恒久対策の設計を人間に提案（日次対応で塞ぎ続けない）。
- 月次の成果物リコンサイル・コスト監査は /ops-review の領分（重複させない）。
