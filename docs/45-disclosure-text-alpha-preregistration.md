# 45 — 開示テキストα 事前登録（候補①：LLM/埋め込みテキスト＝新しいデータ軸）

**作成日**: 2026-06-27
**位置づけ**: docs/03 §6.20 の「9候補独立α 全棄却」を受け、**唯一まだ掘っていない軸＝テキスト**に着手する事前登録。
価格・ファンダ・需給の"再正規化"はもう枯れた（9連敗）。残された次元は EDINET 叙述テキスト。

---

## 0. なぜテキストか（失敗型の構造的回避）

9連敗の失敗型＝直近regime共変／OOS負／符号逆／**momentum代理**／de-risk／古データ減衰／**既試redux**。
これらは全て「価格・ファンダ・需給を別の正規化で測り直した」ことに起因する。**テキストは独立した情報源**で、
価格/ファンダの単調変換では作れない＝momentum代理・既試redux・value共変を**構造的に**避けられる唯一の候補。

### 裏付け（◯=本研究ワークフローで敵対的検証済／△=確立文献・本ワークフロー未検証）
- ◯ **LLM埋め込みのニュース・テキストは横断リターンを予測し、reversal・既知characteristics に対し増分情報**を持ち、
  bag-of-words/Word2Vec/単純センチメントを大きく上回る（Chen, Kelly & Xiu, "Expected Returns and LLMs",
  SSRN 4416687）。効果は**ニュース・アンダーリアクション**＝小型・短ホライズンに集中・大型で消える。
- ◯ **ChronoBERT/ChronoGPT**＝各時点で利用可能なテキストのみで訓練した**時系列整合LLM**＝ルックアヘッドを除去、
  実運用Sharpeは巨大モデル級（arXiv 2502.21206）。＝**当検証ファクトリのPIT規律に噛み合う**。
- ◯ **Ebisu**（日本語金融NLP）＝IR-QAの「暗黙のコミットメント vs 拒否」認識が専門家タスク化（arXiv 2602.01479）
  ＝日本語開示から"センチメント超え"信号が取れる証拠。
- △ **"Lazy Prices"**（Cohen-Malloy-Nguyen, JF 2020）＝**年次報告書テキストの前年比"変化"**（類似度低下＝
  ストーリー書き換え）が将来リターンを**負**に予測。本事前登録の中核設計の直接の先行研究。
- △ Loughran-McDonald 系の金融テキスト分析（語彙・トーン）。

---

## 1. 実現可能性（実地検証済み・2026-06-27）

`data/edinet/` の type=5（XBRL→CSV）コーパスで確認:
- **叙述テキスト実在**: 有報1本に **52–59種のテキストブロック**（`jpcrp_cor:*`、`read_xbrl_csv` の `value` 列）。
  例＝**業績等の概要（MD&A）**・コーポレート・ガバナンスの状況・事業等のリスク・対処すべき課題・セグメント情報。
- **規模**: ダウンロード済み doc zip **53,544本**（うち有報 ~38k）。list に四半期報告書(140)も多数＝将来は四半期拡張可。
- **PIT**: `list/{YYYYMMDD}.parquet` の `submitDateTime`（提出日時＝PITアンカー）。`secCode`(5桁)で J-Quants Code と直結。
- **抽出経路**: `edinet.read_xbrl_csv(zip)` → `item_name` でセクション選別 → `value` が本文。ネット不要・キャッシュ読取。

---

## 2. 事前登録：仮説と設計（**走らせる前に固定**）

### 2.1 主仮説（falsifiable）
**H1（Lazy Prices／テキスト変化）**: 有報のMD&A系叙述テキストの**前年比類似度が低い（＝記述を大きく書き換えた）
銘柄は、その後アンダーパフォームする**。低類似度をショート／高類似度（記述安定）をロングの分位 L/S が、
セクター中立・PITユニバース・大域デフレートDSR で **DSR を有意に上げ、かつ value/quality/PEAD/momentum に
残差化しても残る**。

**H2（局在＝負のコントロール）**: 効果は**小型・情報環境の薄い銘柄に集中**し、**大型で消える**
（limits-to-attention／soft-information 仮説）。大型サブサンプルで効果が同等以上に出るなら、行動仮説は棄却され
"何か別物（regime共変等）"を疑う。

### 2.2 経済的根拠
経営者の叙述（トーン・前向き表現・リスク言及・戦略）には**数値に出ない soft information** がある。不利な実態を
言葉で薄める／物語を書き換える行為は、注意の限界と裁定の限界により**緩やかに織り込まれる**。反対側は叙述を
読まず数値だけ見る参加者。価格/ファンダの単調変換では作れない独立成分。

### 2.3 データ・PIT規律
- **対象テキスト**: 有報の `業績等の概要`／`経営者による財政状態、経営成績及びキャッシュ・フローの状況の分析(MD&A)`／
  `事業等のリスク`／`対処すべき課題`（＝soft・前向き・経営判断セクション。定型の財務注記＝金融商品/退職給付等は除外）。
- **PITアンカー**: `submitDateTime`。リバランス日 t には `submitDateTime ≤ t − lag(既定1営業日)` の最新有報のみ採用。
- **前年比**: 同一 Code の**1期前の有報**テキストと比較（両方とも提出済＝先読みなし）。
- **ユニバース**: 既存 PITユニバース（流動性上位・普通株）。月末リバランス・フォワード月次リターン。

### 2.4 シグナル構築（**段階設計＝最安検証ファースト**）
- **Stage 1（新規依存ゼロ・完全PIT）**: `sklearn.TfidfVectorizer`（文字 n-gram）で各有報MD&Aをベクトル化 →
  **前年比コサイン類似度** = text_similarity。シグナル = text_similarity（高い＝記述安定＝ロング）。
  ＋補助: 文書長変化・新規語比率。＝Lazy Prices の素朴版。**ここで"テキスト変化に edge があるか"を依存ゼロで先に裁く。**
- **Stage 2（埋め込みアップグレード）**: 日本語センテンス埋め込み（候補＝chronologically-consistent or
  pre-2016訓練モデルでリーク最小化／現代モデルは"modest leakage"を明記）で同じYoY類似度＋content射影。
  **Stage 1 を上回るかで「埋め込みが bag-of-words に勝つ」を当データで実証。**

### 2.5 judge_grid 連結（既存機構）
`scope="disclosure_text_change"`、`CrossSectionalStrategy`（q∈{0.1,0.2,0.3}）、セクター中立・winsorize・z、
PITユニバース、執行ラグ1・コスト15bps・ADV容量、永続レジストリで大域デフレートDSR。`examples/research_text_lazy_prices.py`。

### 2.6 合格基準（事前固定）
1. **DSR ≥ 0.95**（多重検定後）。
2. **独立性必須**: value(B/M)・quality(accruals/roe)・PEAD(予想改訂)・momentum に**残差化しても**符号一貫・SR保持。
   `factors.cross_sectional_residualize` で⊥既知factorを取り、残差の両側デフレートDSRを報告。
3. **局在パターン（H2）**: 小型で効き大型で消える。大型で同等以上＝行動仮説棄却＝採用見送り。
4. **サブ期間安定**: 2016-19/2019-23/2023-26 で符号一貫（直近のみ＝④型は不採用）。

### 2.7 失敗型ガード（9連敗の各型を事前に潰す）
| 過去の失敗型 | 本設計での回避 |
|--|--|
| momentum代理 | テキスト変化は価格と独立。**momに残差化**して残るか必須検定 |
| value共変 | **valueに残差化**＋小型局在（H2）で value-regime 共変を切り分け |
| 既試redux | テキスト＝新データ軸。既存factorと低相関を事前確認 |
| OOS負/直近のみ | サブ期間符号一貫を合格条件化、保留OOS報告 |
| リーク | PITテキスト＋Stage1は完全PIT（TF-IDFは当該文書のみ）。Stage2は埋め込みリークを明記・chrono整合を優先 |
| de-risk | 横断L/Sでαを測る（指数タイミングでない） |

---

## 3. 実装ステージ
1. **テキスト抽出基盤** `invest_system/equities/disclosure_text.py`：(a) list から有報のPIT索引（Code,submitDateTime,docID）、
   (b) doc zip から対象セクション本文を抽出。純関数＋オフラインテスト（合成XBRL-CSV）。
2. **Stage 1 研究** `examples/research_text_lazy_prices.py`：YoY TF-IDF類似度 → judge_grid ＋ 独立性残差化 ＋ 局在 ＋ サブ期間。
3. **Stage 2**：埋め込みモデル選定（依存追加の意思決定）→ 同パイプラインで TF-IDF を上回るか。
4. **将来**：四半期報告書(140)・決算短信(TDnet・要追加取得)で頻度を上げ、短ホライズン news 版（Chen-Kelly-Xiu型）へ。

### 埋め込みモデルの意思決定（Stage 2・ユーザー判断事項）
- **A. TF-IDF のみで先行**（推奨・依存ゼロ・完全PIT）＝核心仮説の存在確認。
- **B. sentence-transformers＋日本語モデル**（multilingual-e5 / 日本語FinBERT）＝現代モデルゆえ pre-2024 backtest に
  modest leakage（ChronoBERTが定量化＝致命的でない）。要 torch/transformers 依存。
- **C. chronologically-consistent**（ChronoBERT系・日本語可用性は要確認）＝理想的PITだが入手性リスク。

---

## 4. 出典
- Chen, Kelly, Xiu — Expected Returns and Large Language Models: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687 ◯
- ChronoBERT/ChronoGPT (chronologically consistent LLMs): https://arxiv.org/abs/2502.21206 ◯
- Ebisu 日本語金融NLP: https://arxiv.org/abs/2602.01479 ◯
- Cohen, Malloy, Nguyen — Lazy Prices (JF 2020) △（確立文献・本ワークフロー未検証）
- ECC Analyzer (earnings-call LLM→vol): https://arxiv.org/abs/2404.18470 ◯
