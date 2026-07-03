---
name: research-scout
description: 最新の金融工学・クオンツ・数理統計の文献（学術＋実務）を調査し、この repo のデータ資産と検証規律で裁けるアイデアに翻訳して research_loop/IDEAS.md に備蓄、上位を BACKLOG に昇格する。/research-cycle の供給側。引数でテーマ指定可（例: /research-scout クリプト マイクロストラクチャ）。
---

# /research-scout — 文献調査→IDEAS 備蓄→BACKLOG 昇格（1起動＝1調査）

最新研究から「この repo で検証可能な仮説」を収穫する。**調査と翻訳のみ**を行い、検証は行わない
（検証は `/research-cycle` の仕事）。書き込み先は `research_loop/` のみ。

## ガードレール（違反不可）

1. **K=0**: バックテスト・ローカルリターンの参照・registry への書き込みを一切行わない。
   （データの**存在・スキーマ・件数**の確認は可。リターン・シグナルの計算は不可。）
2. **書き込みは `research_loop/` のみ**（IDEAS.md・surveys/・BACKLOG.md）。docs/・examples/・
   invest_system/・Phase 2 系に触れない。
3. **全アイデアに実在出典必須**: WebFetch で URL/arXiv ID の実在と内容一致を確認できたものだけ
   記録する。確認できない引用は破棄（=捏造引用の構造的排除）。
4. 論文の結果は**他市場・他データでの証拠**として扱う。JP での有効性の主張はしない
   （それを決めるのが research-cycle の事前登録検証）。
5. IDEAS.md は append-only（状態タグのみ更新可）。棄却も理由付きで必ず残す（再浮上防止）。
6. 同時に scout を2つ走らせない。git コミットは実行毎に1回・対象は `research_loop/` のみ。

## 手順

### Step 0 状態ロード
```powershell
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\loop_status.py
```
続けて読む: `research_loop/IDEAS.md`（既存 I-n と出典＝重複回避リスト）・`BACKLOG.md`
（⬜残数・§2 除外リスト・§3 棄却台帳）・`HEURISTICS.md`（H-1〜n）・`LESSONS.md`（F1〜n・
メタ観察）・`docs/24-data-catalog.md` の早見表（何が検証可能か）。
survey_id = `YYYY-MM-DD-<theme>`。

### Step 1 テーマ選定
- 引数があればそれを最優先。
- 無ければ、4領域（a:日本株XS / b:イベント駆動 / c:時系列・マルチアセット / d:クリプト）＋
  方法論枠（検証法・ポートフォリオ構築・ラベリング等）のうち、**BACKLOG/IDEAS が薄い領域**から
  2〜3テーマを選ぶ。LESSONS のメタ観察（例: 2023+ 資本規律レジームで value のみ耐久）に
  照らして「今の地形で生き残りそうな軸」を優先。

### Step 2 fan-out 調査（3〜5並列サブエージェント）
Agent tool（general-purpose）で並列起動。各エージェントに渡すもの:
- 担当テーマ＋探索先: arXiv（q-fin.PM/q-fin.TR/q-fin.ST・stat.ME/stat.ML）直近12〜24ヶ月・
  SSRN・主要ジャーナル（JF/JFE/RFS/Management Science/JPM/Quantitative Finance）・
  実務系（AQR/Man/Robeco/Two Sigma 等のリサーチノート・実務家ブログ・カンファレンス資料）
- **除外リスト**（BACKLOG §2）と**既存 IDEAS の出典 ID**（重複回避）
- **データ資産の要約**（docs/24 早見表の抜粋: J-Quants 日足2016+・財務・信用/空売り・
  EDINET/TDnet/ToSTNeT・外部11資産・bitbank。無いもの: 先物カーブ・ニュース本文・板情報）
- 返却フォーマット（1候補につき）: タイトル／出典（著者・年・arXiv ID or DOI・URL）／
  機構1段落／元市場と主結果（数値）／JP適用に必要なデータ／想定される失敗型（F1〜n から）
- 指示: 「JP 小口個人クオンツが月次〜日次で執行可能なもの」を優先。HFT・板情報・
  オプションMM等の実行不能領域は near-miss として理由付きで報告。

### Step 3 引用実在性チェック
各候補の出典を WebFetch で開き、abstract がエージェントの要約と一致するか確認。
不一致・404・見つからないものは破棄（surveys レポートに「検証不能で破棄」と記録）。

### Step 4 事前スクリーン＆スコア（K=0・ローカルリターン非参照）
各候補に機械的に適用（HEURISTICS 全項目・特に）:
- H-8 データ実在性（docs/24 と突合。無ければ 🗑 か「データ待ち」注記付き 💡）
- H-4 minTRL/MinBTL 概算（想定SR・観測数で認定到達可能か）
- H-3 コスト床（月次XS 15bps／日次イベント 30bps／貸株）・H-12 SR上限概算（イベント系）
- H-1 momentum 代理／H-2 value・PBR改革共変／H-6 独立≠エッジ／H-13 等の prior
- F1〜n のどれで死にそうかを1行で予想（research-cycle の §0 表の下書きになる）
スコア: **機構**（経済的合理性の強さ）・**新規性**（registry 52+ scope・BACKLOG・IDEAS との距離）・
**データ適合**（ローカルで完結するか）・**実装コスト**（低いほど高スコア）各1-5。

### Step 5 記録
- `IDEAS.md` §1 に全候補を append（💡 or 🗑・棄却理由必須・番号は既存最大+1から連番）。
- `research_loop/surveys/<survey_id>.md` に調査レポート: テーマ・検索クエリ・参照した論文
  リスト（見たが拾わなかった near-miss と理由を含む）・所感。

### Step 6 自動昇格（上位1〜2件）
事前スクリーン通過のスコア上位 1〜2 件を `BACKLOG.md` §1 に ⬜ で追記:
- 標準構造（仮説／経済的根拠／データ／新規性／独立性／注意）＋ **出典** 欄（I-n と相互リンク）
- 仮説の**方向**は論文の報告に従って事前固定できるよう明記（他市場証拠＝方向覗き見でない）
- IDEAS 側の状態を ⬆ promoted に更新
- 通過候補が無ければ昇格0件でよい（無理に昇格しない）

### Step 7 コミット＋サマリ
- `git add research_loop/` → `research(scout): survey <theme> — 収穫N件/棄却M件/昇格L件`
- ユーザー向け1画面サマリ: テーマ／参照論文数／IDEAS 追加・棄却・昇格の件数／
  昇格アイデアの1行紹介（出典付き）／次の推奨アクション（/research-cycle 実行 or 追加調査）。

## 運用メモ
- BACKLOG ⬜ が2件を切ったら実行を推奨（月1目安）。
- SSRN 等スクレイプ耐性のあるソースは abstract ページ・Google Scholar 経由で確認できる範囲に
  留める（取れないものは記録しない）。
