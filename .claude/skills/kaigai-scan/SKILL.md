---
name: kaigai-scan
description: 英語圏で日本株がどう語られているかの偵察。SNS/コミュニティ（Reddit・X・StockTwits）、投資メディア（Bloomberg/Reuters/FT/Barron's/Nikkei Asia/Seeking Alpha）、機関投資家シグナル（ファンドマネージャー調査・日本株ETFフロー・13F・アクティビスト動向の英語報道）を3層で収集し、日英の切り口ギャップと「機関が何を根拠に日本株へ資金を動かすか」を分析。ローカルの海外勢実フロー（投資部門別）・大量保有と突合するのが独自機能。裁量レーン＝研究入力禁止。引数: テーマ or 銘柄コード（省略時=全般スキャン）。
---

# /kaigai-scan — 英語圏の日本株談義スキャン（1起動＝1スキャン）

狙いは3つ: ①英語圏で**どの銘柄・テーマ・技術**が話題か ②**日本語圏との切り口ギャップ**
③**機関投資家が日本株ローテーションの根拠に何を使っているか**（公開情報からの推定）。

## ガードレール（違反不可）

1. **エビデンス格付け**: [機関]（FM調査・運用会社レター・規制文書・ETFフロー統計）/
   [報道]（大手メディア・ストラテジストノートの報道）/ [UGC]（Reddit・X・StockTwits・
   Seeking Alpha個人記事）/ [未確認]。全項目に出典URLと日付。UGC 単独を根拠にしない。
2. **機関の判断根拠は「公開情報からの推定」と明記**: 調査（BofA FMS 等）の報道・
   ストラテジストノートの引用・ファンドレターは見えるが、実際の意思決定は非公開。
   インサイダー的知見のように書かない。
3. ペイウォール（FT/Bloomberg 等）は見出し・スニペットまで（回避しない）。
   WebSearch/WebFetch の単発参照のみ＝自動巡回しない。
4. **裁量レーン**: 出力は `output/kaigai/` のみ。SNS/報道は PIT 不能＝研究・レジストリに
   入れない（定型注記）。銘柄の推奨ではなく談義の観測。

## 手順

### Step 0 対象整理
- 引数が銘柄なら英語社名と ADR ティッカーに解決（例: 7203→Toyota/TM。検索で確認）。
- 省略時は全般スキャン＝「今、英語圏の日本株談義で何がホットか」。
- 直近の `output/kaigai/scan_*.md` があれば読み、**前回からの話題の入替**を追う体制で始める。

### Step 1 コミュニティ/SNS層（UGC）
- Reddit: `site:reddit.com` で r/stocks・r/investing・r/ValueInvesting・r/SecurityAnalysis ×
  "Japan stocks / Japanese equities / <ticker>"。直近1ヶ月中心・反応の多いスレ優先。
- X: `site:x.com` で "Japan equities / JPX / <ticker>" ＋日本株ウォッチャー系アカウントの言及。
- StockTwits: 主要 ADR（TM・SONY・MUFG・SMFG・HMC・TAK 等）のストリーム雰囲気。
- 集めるのは「**繰り返し出てくる銘柄名・テーマ・技術**」と熱量（スレ数・反応）。
  単発の煽りは拾わない。

### Step 2 メディア/プロ層（報道）
- Bloomberg・Reuters・FT・WSJ・Barron's・Nikkei Asia の直近の日本株記事
  （WebSearch: "Japan stocks", "Japanese equities", "Topix", "governance reform" 等）。
- セルサイドの日本株ストラテジー（GS/MS/JPM/UBS 等）の**報道された**見解・目標水準。
- Seeking Alpha 等の ADR 個別分析（英語圏の個人〜セミプロがどの銘柄をどう評価するか）。

### Step 3 機関投資家シグナル層（ユーザーの本丸: 資金移動の判断材料）
- **配分調査**: BofA Global Fund Manager Survey の日本株アロケーション（最新月の
  overweight/underweight）の報道。変化の理由として挙げられた項目を列挙。
- **ETF フロー**: EWJ・BBJP・DXJ・HEWJ 等の直近フロー（etf.com・ETFdb・報道）。
  ヘッジ付き(DXJ/HEWJ) vs ヘッジ無しの選好＝円観の代理指標。
- **13F・ファンドレター**: 日本株を積んだ米ファンドの報道・四半期レター
  （バフェット/バークシャーの商社、アクティビスト: Elliott・ValueAct・Dalton・Oasis 等）。
- **ナラティブの棚卸し**: 機関が繰り返し引用する根拠を分類
  （ガバナンス/PBR改革・自社株買い記録・持合い解消・円安の益出し・BOJ 正常化・
  中国代替(alt-China)・半導体装置・現金リッチBS 等）。「どの根拠が新しく増えたか」が主眼。

### Step 4 ローカル突合（この repo 固有・話題と実弾の照合）
インライン python で:
- **海外勢の実フロー**: `flows.section_net_flow(load_investor_types(), investor="foreign")`
  の直近4-8週 ＝ 英語圏の熱と実際の買い越し/売り越しが整合しているか（乖離こそ知見）。
- 話題銘柄のローカル状態: 信用・空売り残高、`edinet/large_holdings*.parquet` の
  外資系保有異動、`jpx_tostnet` 超大口、直近リターン（過熱/出遅れ）。
- **日英ギャップ表**: 「EN で話題だが JP で静か」「JP で話題だが EN に出ない」を各3件まで
  （後者は将来の海外勢フローの候補＝先回り観点）。

### Step 5 出力
`output/kaigai/scan_{YYYYMMDD}.md`:
① ホット銘柄・テーマ表（層別 [機関]/[報道]/[UGC]・前回比の新規/継続/消滅）
② 日英ギャップ表 ③ 機関ナラティブの棚卸し（根拠の増減） ④ 実フローとの整合/乖離
⑤ 裁量メモ（watchlist 候補・/kabu-funda で深掘りすべき銘柄） ⑥ 情報品質表＋定型注記。
ユーザーへは①②③の要点を1画面で。

## 運用メモ
- ナラティブの回転は遅い＝**月1-2回で十分**（BOJ 会合・FMS 公表(月中)・決算シーズン後が好機）。
- 話題銘柄の深掘りは /kabu-funda（Step 4 の SNS 層と接続）、裁量案化は /tactical-brief
  レーンB へ。海外勢フローの定量は /market-brief が毎回出している（そちらが正）。
- ADR の無い中小型が英語圏で話題になっていたら特記（発見価値が高い＝英語圏の
  スモールキャップ発掘系ニュースレターや 5% ルール報告が源泉のことが多い）。
