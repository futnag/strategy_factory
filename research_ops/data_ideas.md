# DATA IDEAS — オルタナ/新規データソース台帳（/data-scout の備蓄層）

**運用ルール**: `D-n` 番号制・append-only（状態タグのみ更新可）。全エントリに検証可能な
URL・ライセンス評価・PIT 評価を必須。上位は「取得待ち行列」に昇格し `/data-acquire` が実行。
**状態タグ**: 💡candidate ／ ⬆acquire-queued（取得待ち行列入り）／ ✅acquired（取得済み）／
🗑rejected（理由必須）

**取得待ち行列（優先度順・/data-acquire の入力）**:
1. D-3 BTC 現物ETFフロー日次（IDEAS I-48 解禁）
2. D-4 優待データベース（I-31/I-42 の2案解禁）
3. D-5 目論見書ロックアップ条項（I-41 解禁・パース build 含む）
4. D-7 TSE 資本コスト開示一覧 **2024-01〜2025-04 の archive backfill**（governance_event_value の salience shock 窓・D-2 の続き）
5. D-8 EDINET「自己株券買付状況報告書」2016-2024 backfill（buyback_execution_flow を powered 化・I-51）
- ✅ 取得済: D-1（TOPIX 削減リスト）・**D-2（TSE 資本コスト開示一覧・2025-05〜・2026-07-05）**

---

## §1 台帳

### D-1 ✅ JPX TOPIX 段階的ウエイト低減リスト
- **取得済み 2026-07-04**（/data-acquire E2E・research_ops/acquisitions/2026-07-04-jpx-topix-lists.md）。
  `data/jpx_indices/`・BACKLOG ⬜ topix_staged_flow を解禁。第2段階リスト（2026-08 基準日以降
  公表）は前向き追加待ち。

### D-2 ✅ 東証「資本コストや株価を意識した経営」開示企業一覧（月次）
- **取得済 2026-07-05**（/data-acquire・`data/tse_capital_disclosure/`・記録 research_ops/acquisitions/2026-07-05-tse-capital-disclosure.md）。
  list.xlsx から**月末snapshot 13断面（2025-05〜2026-05）**を PIT パネル化・governance_event_value を ⬜ 解禁。
  **限界**: list.xlsx は ~13月ローリング＝**2024 salience shock は未収録**＝**D-7（archive backfill）**へ分離。
- **出所**: JPX 公表アーカイブ（月次 Excel/PDF・2024-01〜）。**ライセンス**: 公表資料・
  ローカル保持のみ。**PIT**: 公表月がアンカー＝一覧の月次スナップショット系列として再構成可
  （現在の一覧から過去を推定するのは PIT 偽装＝不可・必ず各月の原本を取る）。
- **解禁**: research_loop/BACKLOG ⏸ governance_event_value（JCF forthcoming の出典アンカー付き・
  Standard 市場は開示~50%＝イベント継続中）。**履歴深度**: 2024-01〜＝2.5年・月次。
- **スコア**: 解禁価値4・PIT堅牢性4（原本アーカイブが残っているかの確認が先決）・
  取得コストS-M・維持コストS（月次1ファイル）。

### D-3 ⬆ BTC 現物ETF 日次フロー
- **出所**: 発行体の日次開示の集計（Farside Investors 等の無料トラッカー or 発行体公表の突合）。
  **ライセンス**: トラッカーの利用規約要確認（自前で発行体開示から再構成すれば堅牢）。
- **PIT**: フローの速報/確報の区別が本丸（米夕方公表＝JST 朝に既知・I-48 の設計条件）。
- **解禁**: IDEAS I-48。**履歴深度**: 2024-01〜。
- **スコア**: 解禁価値3・PIT堅牢性3・取得コストS・維持コストM（日次追記）。

### D-4 ⬆ 株主優待データベース（銘柄×優待内容×価値×権利月）
- **出所候補**: 公開の優待情報サイト（スクレイプ可否の ToS 確認が先決）or 書籍/公表資料からの
  構造化。**PIT**: 導入/変更/廃止の**発表日**が必要＝TDnet 前向き蓄積との突合で将来分は堅牢・
  過去分は現在DBからの逆算になりがち（PIT 偽装リスク＝過去分は「水準特性」用途に限定する等の
  設計制約を D-n に明記して運用）。
- **解禁**: I-31（優待プレミアム）・I-42（権利日ランアップ）。
- **スコア**: 解禁価値3・PIT堅牢性2（過去分）・取得コストM・維持コストM。

### D-5 ⬆ IPO 目論見書のロックアップ条項（対象株主・期間・1.5倍トリガー）
- **出所**: EDINET 有価証券届出書の本文（ローカル metadata あり・本文パースの build）。
  **ライセンス**: EDINET 公表書類＝問題なし。**PIT**: 上場前公表＝堅牢。
- **解禁**: I-41（ロックアップ解除・time-only×VC重コホート）。
- **スコア**: 解禁価値3・PIT堅牢性5・取得コストM-L（テキスト抽出 build）・維持コストS。

### D-6 💡 JSF 貸借取引データの過去分（逆日歩・貸株残）
- **出所**: 日証金の公表データ（`examples/update_jsf.py` が前向き取得の器として存在・
  `data/jsf/` は未蓄積）。過去分のアーカイブ可否は未調査。
- **解禁**: docs/48 limit_reversal の再評価条件・ショート系全般のコスト実値化（H-3 の精緻化）。
- **スコア**: 解禁価値3（低EV案件の解禁が主）・PIT堅牢性4・取得コストS（前向き）〜?（過去分）。

### D-7 ⬆ TSE 資本コスト開示一覧の archive backfill（2024-01〜2025-04）
- **出所**: JPX 過去公表（news 添付 PDF・月次 list の旧版）or Wayback Machine の list.xlsx 履歴。
  **PIT**: 各月末原本のスナップショット（現行 D-2 と同スキーマ化）。**ライセンス**: 公表資料・ローカル保持のみ。
- **解禁**: governance_event_value の **2024 salience shock 窓**（I-9 D'Ercole の本丸）。現行 D-2 は 2025-05+ のみで
  Prime の大規模開示イベント（2024）を左側打切り。**フォーマット改定（2025-01/09 見直し）で列定義が異なる可能性**＝要検証。
- **スコア**: 解禁価値4・PIT堅牢性3（旧版原本の入手可否が先決）・取得コストM・維持コストS。

### D-8 ⬆ EDINET「自己株券買付状況報告書」2016-2024 backfill
- **出所**: EDINET API（doc type 指定で日次再取得）。ローカル `edinet/list/` は 2025-01+ のみこの doc type を収録
  （2016-2024 は 0＝バックフィル除外）。**ライセンス**: EDINET 公表＝問題なし。**PIT**: submitDateTime アンカー＝堅牢。
- **解禁**: buyback_execution_flow（I-51）を **~18月→~10年に powered 化**（現状は 2025+ の現レジーム内のみで F2/H-4）。
- **スコア**: 解禁価値4・PIT堅牢性5・取得コストM（~2200日 API 再取得）・維持コストS（夜間で追随）。

（以降、/data-scout が追記）
