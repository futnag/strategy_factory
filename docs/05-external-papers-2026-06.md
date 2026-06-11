# 05 — 外部論文リファレンス(2026-06 取込・検証済み)

**作成日**: 2026-06-11 / **作成者**: Claude (claude.ai セッション)
**経緯**: ChatGPT・Gemini・Grok に「個人システムトレードに応用できる最新論文」を調査させた出力と、Claude 独自の web 調査を統合。**全論文について arXiv 等で実在・書誌情報を検証済み**。
**用途**: `docs/04-paper-integration-plan-2026-06.md` のギャップ分析の根拠資料。**採用候補・採用根拠となりうる文献のみを収録**(不採用・参照のみと判定したものは削除済み)。エントリ ID(A1, C3 等)は 04 からの参照を維持するため欠番を許容している。

---

## A. 検証・過学習対策

### A1. The GT-Score: A Robust Objective Function for Reducing Overfitting in Data-Driven Trading Strategies
- arXiv:2602.00080 (2026-01) / https://arxiv.org/abs/2602.00080
- 概要: リターン+統計的有意性+一貫性+ダウンサイドリスクの複合目的関数。事後補正(DSR)と異なり、探索段階から過学習しにくい領域へ誘導する。補足資料に再現コードあり。
- 本リポジトリとの関係: 判定哲学は DSR で実装済み。GT-Score は「グリッド内の脆い構成の早期識別」という補助用途で導入余地(→ 04 P2-B)。

### A2. Implementation Risk in Portfolio Backtesting: A Previously Unquantified Source of Error
- arXiv:2603.20319 (2026-03) / https://arxiv.org/abs/2603.20319
- 概要: 同一戦略でもバックテストエンジンが違えば結果が変わる「実装リスク」を定量化。統計的過学習と独立した誤差源であり、多重検定補正を全てパスした戦略でもエンジン次第で Sharpe が大きく変わりうる。
- 本リポジトリとの関係: **未対応のギャップ**。単一自作エンジン(277テスト)だが独立実装との突合は未実施。Phase 2 照合で実バグ2件を捕獲した実績(03 §10)がこの論文の主張を裏付ける(→ 04 P1-A)。

## B. レジーム適応

### B3. DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management
- arXiv:2601.05975 (2026-01) / https://arxiv.org/abs/2601.05975
- 概要: コスト控除後リスク調整リターンを直接損失関数にする decision-focused 学習。2段階 MVO 比約2倍のネット性能、回転率を自然に抑制。
- 本リポジトリとの関係: 深層 end-to-end は標本制約(月次≈120)から取り込まないが、「ネット成果を直接最適化」「回転率抑制」の思想から**回転率抑制の安価な代替=リバランス・デッドバンド**を導出(→ 04 P2-A)。

## C. クロスセクショナル戦略

### C1. Building Cross-Sectional Systematic Strategies By Learning to Rank — Poh, Lim, Zohren, Roberts
- arXiv:2012.07149 (2020-12・本テーマ原典) / https://arxiv.org/abs/2012.07149
- 概要: 銘柄選択を「リターン予測→ソート」でなく LTR(LambdaMART 等)で順位を直接学習。従来比で Sharpe 大幅改善・DD 耐性向上。

### C2. Learning to Rank: Enhancing Momentum Strategies Across Asset Classes — Burdorf
- SSRN 5255258 (2025-05-15) / https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5255258
- 概要: C1 をマルチアセットへ拡張、複数アセットクラスで有効性を確認。
- C1/C2 と本リポジトリの関係: **未検証の新アーキタイプ**。既存ファクター(value/quality/momentum/low-vol/accruals)を特徴量とした順位学習は、素朴ランク(§6.4/6.7)の上位互換仮説として検証価値あり(→ 04 P3-A)。日本はモメンタム負(§6.7)のため結果は自明でない=genuine question。

### C3. ML Enhanced Multi-Factor Quantitative Trading with Bias Correction — Yimin Du
- arXiv:2507.07107 (2025-06-02, v2 2026-05-09) / https://arxiv.org/abs/2507.07107 / コード(MIT): https://github.com/initial-d/ml-quant-trading
- 概要: 値幅制限による約定不能価格がローリング・ファクター計算に混入する「**上流汚染(upstream contamination)**」を指摘。A株実データで見かけ IC を18%水増し・実現 Sharpe を0.44pt 毀損。データ読込時点で約定可能性マスクを全演算子に通す **mask-first 設計**で解決。
- 本リポジトリとの関係: **執行面(DP15: no_buy/no_sell キャリー)は実装済みだが、特徴量計算面は未監査**。momentum_12_1/vol_20/reversal_5 等が張り付き日価格を素通しで計算している可能性(→ 04 P1-B)。

## D. ポートフォリオ・リスク管理

### D2. Deep RL for Optimal Portfolio Allocation: Comparative Study with MVO — Sood, Papasotiriou, Vaiciulis, Balch (J.P. Morgan)
- FinPlan'23 (ICAPS 2023) / https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf
- 概要: 同一目標・コスト込みで DRL と MVO を公平比較。MVO は推定誤差への過剰反応で回転率コスト負け、RL は「取引しない我慢」を学習。
- 本リポジトリとの関係: 教訓「**不要な取引の抑制が公平比較での勝因**」はデッドバンド導入の傍証(→ 04 P2-A)。

## E. LLM 系

### E3. Adversarial News and Lost Profits — Rizvani, Apruzzese, Laskov (IEEE SaTML 2026)
- arXiv:2601.13082 (2026-01-19) / https://arxiv.org/abs/2601.13082
- 人間に不可視・LLM には可読の敵対的ニュース(Unicode ホモグリフ・不可視テキスト)で、単日攻撃により累積リターン平均約3.5%低下。利益は出続けるため運用者が気づかない。
- 本リポジトリとの関係: 現状ニュース/NLP 不使用のため非該当。**将来ニュース層を追加する場合の必須設計原則**(取込時 NFKC 正規化+不可視文字パージ)として予約(→ 04 P3-B)。

## F. コスト現実主義

### F1. ML-Based Bitcoin Trading Under Transaction Costs: Walk-Forward Forecasting — Bysik, Ślepaczuk
- arXiv:2606.00060 (2026-05-19) / https://arxiv.org/abs/2606.00060
- 概要: BTC/USDT 時間足・27フォールド walk-forward。コスト10bps で素朴な符号戦略は全滅。**予測の大きさがコスト閾値を超えた時のみ取引する cost-aware 実行フィルター**で回転率激減→収益性回復(ロングオンリー XGBoost で Sharpe>1)。
- 本リポジトリとの関係: コスト控除・ボラ連動コストは実装済みだが「**閾値以下の取引をしない**」機構は未実装(→ 04 P2-A デッドバンド)。

## G. Factor Zoo・アノマリー統合(2026-06-11 追加・全件検証済み)

### G1. The Co-Pricing Factor Zoo — Dickerson, Julliard, Mueller
- SSRN 4589786 (原型 2023-10) / arXiv:2604.04430 / **Journal of Financial Economics 採択** / https://arxiv.org/abs/2604.04430
- 概要: 株式+社債の同時価格付けで18 quadrillion個のモデル空間を Bayesian Model Averaging で分析。少数の頑健ファクター+「真の SDF は観測ファクター空間で dense(多数の弱いノイズ付きプロキシの集合)」という構造を示し、BMA-SDF が全ての低次元モデルを in/out-of-sample で上回る(OOS Sharpe 1.5-1.8)。
- 留保: Sharpe は SDF レベルの理論値(非取引可能ファクター含む・コスト前)。複製可能な運用戦略の数字ではない。原型は2023年。
- 本リポジトリとの関係: 実務的エッセンスは「**最良モデルの選択より、モデル群の平均が勝つ**」。judge_grid の best-of-grid 選択バイアス(§6.4 gap 例)への構造的対策として「グリッド平均シグナル」へ翻訳可能(→ 04 P2-C)。§6.11 の「最良セルを選ばず分布で報告」と同じ哲学の戦略版。

### G2. Enhancing Stock Market Anomalies with Machine Learning — Azevedo & Hoegner
- SSRN 3752741 / Review of Quantitative Finance and Accounting (2023) / https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3752741
- 概要: 299アノマリー × 30種以上のML(250超モデル)・5億 firm-month(米国)。月次 OOS 1.8-2.0%、80%超のモデルが線形ベースライン以上。往復コスト2%まで・公開後アノマリー限定でも頑健と報告。非線形MLがリスクで説明しにくい mispricing を捉える可能性。
- 反証の併読が必須: Avramov et al. は経済的制約(流動性等)を課すと ML 強化アノマリーのアルファが減衰すると報告。Azevedo-Hoegner-Velikov の後続(SSRN 4702406)はコスト・公開後減衰の共同インパクトをより悲観的に定量化。
- 本リポジトリとの関係: **P3-A(LTR ウェーブ)の事前確率を押し上げる独立証拠**(線形合成の失敗 §8 と矛盾しない=失敗したのは線形)。新タスクではなく P3-A の根拠に追記。米国・小型株含むユニバースの振幅を PIT上位300 の日本株に期待しないこと。

### G5. Timing the Factor Zoo — Neuhierl, Randl, Reschenhofer, Zechner
- SSRN 4376898 (2023)
- 概要: 300超ファクターで、過去ファクターリターンとボラが最良の個別予測子。タイミングで中央値+2%/年。
- 本リポジトリとの関係: **実装が軽く本リポジトリに適合的**(→ 04 P3-D 診断)。

---

## I. テキストマイニング系(2026-06-11 追加)

### I1. Predicting Returns with Text Data — Ke, Kelly, Xiu (SESTM)
- 添付PDF (2020-10-10) / 著者: Zheng Tracy Ke (Harvard), Bryan Kelly (Yale・AQR・NBER), Dacheng Xiu (Chicago Booth)
- **品質**: 一流著者・厳密な統計理論(スクリーニング一致性・誤差限界・ランク相関一致性を証明)・大規模実証。テキストマイニング系では最も信頼できる部類。
- 手法(SESTM = Sentiment Extraction via Screening and Topic Modeling): 完全に white-box な3ステップ。(1) 相関スクリーニングで「センチメント帯電語」を選別(リターン符号との共起頻度 f_j を閾値と比較)、(2) 教師ありトピックモデルで各語に正/負トーンを付与(リターンの標準化ランクを教師信号に)、(3) ペナルティ付き MLE で新記事をスコア化。辞書(LM・Harvard-IV)も深層学習も使わず、ラップトップで数分。
- 実証結果(Dow Jones Newswires・1989-2017): 翌日 L-S の EW Sharpe 4.29(VW 1.33)。**ほぼ全てアルファ**(FF5+MOM 後 R²≤10%)。RavenPack(商用)を EW Sharpe で32%上回る。コスト後も EWCT(回転率制約+指数減衰ウェイト)で γ=0.5 のとき純 Sharpe 2.3。
- **数字を文脈化する上での重要な留保**:
  1. EW L-S 4.29 は**小型株が主役**。論文自身が「センチメントは小型株のリターンをより強く予測」と明記(VW では 1.33 に低下)。価格反応は小型株で初動15分に52bps・大型株は11bpsで1日で完了。
  2. グロスのデイリー回転率 約95%。生 Sharpe は回転率制約で大きく削られ、純 Sharpe 2.3 は γ チューニング後の最良。コスト10bps 前提(米大型資産運用者基準、Frazzini et al. 2018)。
  3. day−1 戦略の Sharpe 5.88・day0 が 10.78 は **infeasible**(著者が明記。約定不能なリードラグ相関を経済単位に変換しただけ)。実行可能なのは day+1 のみ。
  4. 効果は **fresh news に集中**(新規性0.75以上で初動39bps vs stale 23bps)。stale news は2日・fresh は4日で織り込み完了=**短期しか効かない**。
- 本リポジトリとの関係: 詳細は 04 P3-F。**手法自体は健全だが、(a) 日本語ニュースの構造化データ基盤が無い (b) 効果が小型株・短期・高回転に集中し本リポジトリの月次・上位300・大型寄りユニバースと相性が悪い** という二重のギャップ。条件付き予約に留める。

---

## J. 著者系譜・AQR/Booth 調査(2026-06-11 追加・SESTM 著者を起点とした広域調査)

**経緯**: SESTM(I1)の著者 Bryan Kelly・Dacheng Xiu を起点に、その研究系譜と AQR・Booth の関連研究を広く調査。全件実在確認済み。

### J1. IPCA(Instrumented Principal Components Analysis)— Kelly, Pruitt, Su
- SSRN 2983919 / NBER w24540 / JFE 2019 "Characteristics are Covariances" / **公開Python実装: github.com/bkelly-lab/ipca**
- 概要: 観測可能な企業 characteristic を「動的ローディングの操作変数(instrument)」として導入し、潜在ファクターと時変ローディングを許す。**characteristic 効果を「リスク補償(潜在ファクターへのエクスポージャー)」か「アノマリー切片」かに構造的に切り分ける**。四つの IPCA ファクターが既存モデルより有意に正確に横断面を説明し、アノマリー切片を小さくする。
- 実装適性: マッピング行列 Γ のサイズを L で制御し L≪N なので、PCA が過学習する大規模横断面パネルを扱え、不均衡パネルにも対応。numpy ベース。本リポジトリの PIT 上位300・約120ヶ月でも L を絞れば適用可能性あり。
- 本リポジトリとの関係: **最も取り入れる価値が高い**。P3-A(LTR)と並ぶ「複数ファクター統合」候補だが、LTR が順位を学習するのに対し IPCA は「リスクとアノマリーを切り分ける」。value 単独耐久性(§7.1)・機械的合成の失敗(§8)と整合的な検証が可能。→ 04 P3-G として独立タスク化。

### J2. AQR 公開データセット(無料)
- aqr.com/Insights/Datasets
- "Value and Momentum Everywhere"(Asness-Moskowitz-Pedersen, JF 2013)の8市場・資産クラスの value/momentum ファクター月次データ、"Trends Everywhere" の時系列モメンタム Sharpe データ(1985-2017)を無料公開。
- 本リポジトリとの関係: 外部11資産(investers/)の拡充・TSMOM オーバーレイ(旗艦第2スリーブ)の外部ベンチマーク照合に直接使える。SESTM で問題になったデータ基盤の課題を回避できる数少ない経路。**ライセンス確認は要**。

### J3. Factor Momentum — Gupta & Kelly(AQR)
- "Factor Momentum Everywhere"(Journal of Portfolio Management)
- 概要: 最近アウトパフォームしたファクター・ポートフォリオは将来も好調、アンダーパフォームしたものは苦戦が続く「ファクター・モメンタム」が、あらゆるファクターベースのポートフォリオで観察される極めて普遍的な現象。
- 但し書き: Asness 自身は"Factor Timing is Hard"(2017)でバリュエーションベースのファクター・タイミングには慎重。有効なのは「モメンタムによるタイミング」に限るという限定付き。
- 本リポジトリとの関係: **P3-D(ファクター・モメンタム診断、G5 由来)の事前確率を独立な一流チームの証拠で補強**。P3-D の根拠欄に追記。

---

## K. 注目組織(AQR/Booth 外・2026-06-11 追加)

**経緯**: 「AQR・Booth 以外に注目すべき組織はあるか」の調査。全件実在確認済み。

### K1. ADIA Lab(Marcos López de Prado)— 本リポジトリの「思想的本家」の最新作
- López de Prado は現在 ADIA(アブダビ投資庁)の Quantitative R&D グローバルヘッド、ADIA Lab 創設ボードメンバー。本リポジトリの DSR/CPCV/PBO は全て彼の手法(=思想的本家)だが、今も最前線で新作を出している。
- **最重要: "Why Has Factor Investing Failed?: The Role of Specification Errors"**(López de Prado & Zoonekynd, SSRN 4697929, 2024-01・2025-11改訂)
  - ファクター投資の偽陽性の原因として、よく理解された p-hacking とは**別の、ほとんど研究されていない原因—計量経済学の正統が奨励するモデル特定の選択**を指摘。リスクプレミアムが一定で正しい符号で推定されても、ファクター戦略がアンダーパフォームし系統的損失を生みうることを証明。
  - p-hacking 駆動のファクター動物園(ノイズをシグナルと取り違え)とは別の現象—**「factor mirage(ファクター蜃気樓)」**—を特定。正統的な計量経済学の慣行が、統計的に強く見えるが構造的に欠陥のある誤特定モデルを系統的に報酬する。
  - **本リポジトリとの関係**: §9 で自認する「在サンプル精緻化が p-hacking 隣接」に、**第3の偽陽性軸(モデル特定の誤り)**を加える。選択バイアス(DSR)・先読み(AsOfView)は潰し済みだが、「ファクターモデルの特定設計そのものが歪んでいる」軸は明示的には未処理。→ 04 P1-A/P1-B の「第3の監査軸」として紐付け。

### K2. Global Factor Data(Jensen-Kelly-Pedersen)— 日本を含む無料データ+コード
- "Is There a Replication Crisis in Finance?"(JF 2023, 78(5):2465-2518)/ SSRN 3774514 / NBER w28432 / **コード・データ公開: github.com/bkelly-lab/ReplicationCrisis**
- 概要: ベイズファクター再現モデルで、大多数の資産価格ファクターが (1) 再現可能 (2) 13テーマにクラスタ化できその大半が接点ポートフォリオの重要部分 (3) **93カ国(日本を含む)の新規大規模データで OOS でも機能** (4) 観測ファクター数の多さで証拠が強まる(弱まらない)と結論。
- 本リポジトリとの関係: **データ基盤の課題(SESTM で多発)を部分的に解決**。(a) J-Quants 自前ファクターを独立な第三者データで突合できる(P1-A の実装リスク検証を日本ファクターで補強) (b) 13テーマクラスタリングは「value 単独が耐久」(§7.1)の外部検証。→ 04 P1-A の外部突合データ源として紐付け。**ライセンス(CC-BY)確認済・WRDS アクセスは要**。

### K4. Shrinking the Cross-Section(Kozak-Nagel-Santosh, U.Maryland/Chicago)— dense SDF の理論
- JFE 2020, 135(2):271-292 / **replication コード公開**
- 概要: 多数の横断面予測子の共同説明力を要約する頑健な SDF を、低分散主成分の寄与を縮小する経済的事前分布で構築。**characteristic スパースな SDF は存在せず、多くのアノマリーが OOS R² に実質的限界貢献するが、PC スパースな SDF はアノマリー・ポートフォリオを良く価格づける**。
- 本リポジトリとの関係: G1(Co-Pricingの dense SDF)と同じ思想で、**P2-C(グリッド平均)の理論裏付けを強める**。「個々の characteristic ではスパース化できないが PC 空間ではできる」は、本リポジトリの合成・縮小設計への示唆。将来 P2-C 拡張時の参照先。

### 調査した主要組織・人物マップ(今後のフォロー先)
| 組織/人物 | 着目点 | 無料リソース |
|---|---|---|
| **ADIA Lab / López de Prado** | 本リポジトリの本家・factor mirage 等新作継続 | SSRN(10位以内の被読著者) |
| **bkelly-lab(Yale, Kelly 研)** | IPCA・ReplicationCrisis・Global Factor Data | GitHub コード+データ |
| **Chen & Zimmermann** | Open Source Asset Pricing(アノマリー再現) | openassetpricing.com |
| **Stanford(Pelger)** | SDF 推定・Missing Financial Data 処理 | mpelger.people.stanford.edu |
| **U.Maryland(Kozak)/Nagel** | Shrinking the Cross-Section・Interpreting Factor Models | replication コード |
| **AQR 公開データ**(J2) | VME/Trends ファクター月次 | aqr.com/Insights/Datasets |

---

## L. 追加調査: 日本で機能する対案・無料データ・旗艦の前提(2026-06-11 追加)

**経緯**: A〜K と重複しない軸(旗艦が依存する仮定の基盤と反証・日本で機能するモメンタムの対案・無料データ源)の調査。全件実在確認済み。

### L1. 残差(idiosyncratic)モメンタム — 日本で機能するモメンタムの複数独立証拠
- 原典: Blitz, Huij, Martens "Residual Momentum"(J. Empirical Finance 2011)/ Chaves "Idiosyncratic Momentum: U.S. and International Evidence"(J. Investing 2016, 25(2):64-76、WP題名 "Eureka! A Momentum Strategy that Also Works in Japan")/ Blitz, Hanauer, Vidojevic "The idiosyncratic momentum anomaly"(IREF 2020, 69:932-957)/ Chang, Ko, Nakano, Rhee "Residual momentum in Japan"(2018)
- 概要: 過去リターンそのものでなく、市場回帰(または FF モデル)の**残差**でソートするモメンタム。市場ベータ起因成分を除去するとボラが下がり、伝統的モメンタムをコントロールしてもアルファが残る。21カ国+米国で確認され、**伝統的モメンタムが無効な日本でも成立**(Chaves)。日本専門の独立追試(Chang et al. 2018・アンダーリアクション説を支持)・国際追試(Blitz et al. 2020: 欧州・日本・アジア太平洋・新興国で頑健)あり。Chaves は「市場モデル(単回帰)の残差化だけで主要な便益が出る」と報告=実装が軽い。
- 本リポジトリとの関係: §6.7 で「日本はモメンタム単独が負」を自前実証済みだが、**残差化という単純な変換で蘇るかは未検証の genuine question**。momentum_12_1 特徴量は既存・残差化は numpy で軽量。複数の独立査読済み証拠が事前確率を押し上げる。→ 04 P3-H として事前登録。

### L2. 日本株の無料データソース(P1-A 外部突合の最短経路)
- **Kenneth French Data Library 日本ファクター**: Fama/French Japanese 3 Factors・Japanese Momentum Factor の月次系列が無料・CSV 直ダウンロード(mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)。**WRDS 不要**で K2(Global Factor Data)より入手障壁が低い。自前 value ファクターとの符号・相関突合の第一候補。**→ 2026-06 実施済み**(03 §6.20: value vs HML corr+0.77・mom vs WML corr+0.74＝構築バグ兆候なし・`examples/verify_factor_external.py`)。なお JKP 公開ポータル jkpfactors.com は国別ファクターリターンを無料配布しているが JS 駆動で CLI 直取得不可(手動 DL は可＝将来の追加突合源)。
- **EDINET API**(金融庁公式・無料): 有報・四半期・大量保有の XBRL を機械取得(API 仕様書 Version 2 公開)。J-GAAP/IFRS 対応の型付き OSS パーサー(edinet-tools 等)あり。**J-Quants fins データの独立突合源**(XBRL 原文から)。提出タイムスタンプがあり PIT 設計に乗る。将来 P3-F(テキスト)のコーパス候補でもある。
- 本リポジトリとの関係: → 04 P1-A の外部突合データ源(K2 と並記、French を先に試す)。

### L3. Anomalies Across the Globe — Jacobs & Müller(公開後減衰の国際的ニュアンス・P3-H の事前確率根拠)
- JFE 2020, 135(1):213-230 / SSRN 2816490
- 概要: McLean-Pontiff(JF 2016、公開後減衰の本家)を241アノマリー×39市場へ拡張。200万超のアノマリー国月で、**公開後減衰が信頼できるのは米国だけ**(EW 62%/VW 66%減)。38の国際市場では信頼できる減衰なし。裁定障壁が市場を分断し、アノマリーはデータマイニングでなくミスプライシングを示唆。
- 本リポジトリとの関係: §6.4/6.14 の「有名な歪みは死ぬ」と矛盾せず精密化する。本リポジトリが FAIL を確認したのは**イベント駆動系**(構造変化を実データで特定済み)であり、Jacobs-Müller が「国際で残る」のは**クロスセクション系**の平均。「イベント系は構造変化で死に、クロスセクション系は日本では裁定資本が薄く残りやすい」という整理は value 耐久(§7.1)と整合し、**P3-H(残差モメンタム)・P3-A/P3-G のクロスセクション系仮説の事前確率を支える**。

### L4. Zeroing In on the Expected Returns of Anomalies — Chen & Velikov(コスト後の決定版・P2-C の根拠補強)
- JFQA 2023, 58(3):968-1004 / SSRN 3073681
- 概要: 204アノマリーを実効スプレッド+公開後+2005年以降で評価。**平均アノマリーの期待ネットリターンは月 4bps**。最強でも 10bps、**複数アノマリーの結合手法で約 20bps**。公開後減衰はグロスで約50%、2005年以降に限ると72%、コスト後は93%。
- 本リポジトリとの関係: 「単一アノマリー追跡でなく合成+コストモデル重視」という設計(DP15/17)の最強の学術的裏付け。**結合が最良(20bps)という結果は P2-C(グリッド平均)の根拠を G1/K4 と独立に補強**。

### L5. On the Performance of Volatility-Managed Portfolios — Cederburg, O'Doherty, Wang, Yan(旗艦の walk-forward 割当設計の裏付け)
- JFE 2020, 138(1):95-117 / SSRN 3357038
- 概要: Moreira-Muir(JF 2017・高ボラ時に露出減でアルファ)のボラ管理を103戦略で包括検証。spanning regression のアルファは再現するが、**含意される戦略はリアルタイム実装不能で、OOS では unmanaged への単純投資に劣る**。原因は spanning regression の**構造不安定**。Barroso-Detzel はコスト後でも生き残らないと報告。
- 本リポジトリとの関係: 旗艦のボラレジーム条件付き switch は直撃を受けない(連続ボラスケーリングでなく、レジーム別スリーブ切替+walk-forward 割当 §6.10-6.11)。むしろ「構造不安定」という診断は**「静的関係を信じず walk-forward で割り当てる」という既存設計判断の根拠を補強**する。将来 §6.9-6.11 を再評価する際の基準点。
