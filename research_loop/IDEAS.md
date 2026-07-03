# IDEAS — 文献調査由来のアイデア台帳（scout の備蓄層）

**運用ルール**: `I-n` 番号制・append-only（**状態タグのみ**更新可・本文の書換禁止）。
全エントリに**検証可能な出典**（arXiv ID・DOI・URL）必須＝捏造引用の構造的排除。
`/research-scout` が追記し、事前スクリーン通過の上位を `BACKLOG.md` §1 へ昇格する。
棄却も理由付きで残す（再浮上防止の dedupe。BACKLOG §2/§3 と合わせて3つの dedupe 面）。
50件を超えたら 🗑/🧪 をアーカイブ節へ移してよい（本文は削らない）。

**状態タグ**（見出し `### I-n <タグ> <タイトル>` — loop_status.py がパース）:
💡 candidate（備蓄）／⬆ promoted（BACKLOG 昇格済み）／🗑 rejected（棄却・理由必須）／
🧪 tested（検証済み・scope リンク必須）

**エントリ項目**: 出典／機構（1段落）／元市場と主結果／JP適用のデータマッピング（docs/24 準拠）／
事前スクリーン（H-n・F-n 参照）／スコア（機構・新規性・データ適合・実装コスト、各1-5）／
survey 参照／登録日。

---

## §1 台帳

### I-1 💡 IPCA／autoencoder 型の条件付きファクターモデル
- **出典**: Kelly, Pruitt & Su (2019) "Characteristics are covariances" JFE 134(3) / Gu, Kelly & Xiu
  (2021) "Autoencoder asset pricing models" Journal of Econometrics 222(1)。
- **機構**: 特性→ファクター露出を時変写像で学習（IPCA は線形・AE は非線形）。静的な GKX 予測 ML と
  違い「特性は共分散（リスク露出）の代理」という価格付け構造を推定する。
- **元市場と主結果**: 米国 XS。IPCA は観測ファクターモデルを OOS で上回る。
- **JPデータ**: `data/phase4/`（GKX 行列・60特徴）をそのまま流用可能。
- **事前スクリーン**: H-11（複雑度 vs 小データ）が正面リスク＝AE は縮退見込み・IPCA（線形）優先。
  conditional_sdf の H1 収穫（価格付け目的>予測）と整合する方向。データ適合は高い。
- **スコア**: 機構4・新規性3（SDF 隣接だが推定枠が別）・データ適合5・実装コスト3。
- **survey**: 2026-06-27 deep-research 調査の未確定分（TODO.md フロンティア節）。登録 2026-07-03。

### I-2 💡 news 埋め込みテキストα（Chen–Kelly–Xiu 型）
- **出典**: Chen, Kelly & Xiu (2023) "Expected returns and large language models" (SSRN 4416687)。
- **機構**: ニュース本文の LLM 埋め込みから期待リターンを構成。開示「変化」でなくニュース
  「内容」を使う＝docs/45 の cheap-text 失敗（年次変化極小・飽和）を構造的に回避する別軸。
- **元市場と主結果**: 米国・グローバル。ニュース埋め込みは強い短期予測力（論文報告）。
- **JPデータ**: **不足**＝TDnet 本文 or ニュースフィードの取得＋埋め込み基盤の大投資が必要
  （`data/tdnet/` は一覧のみ・本文なし）。
- **事前スクリーン**: H-8（データ制約）で当面保留。docs/45 §5 の判断（保留）を継承。
  TDnet 本文パイプライン（docs/47 の拡張）が整備されたら昇格を再検討。
- **スコア**: 機構4・新規性4・データ適合1・実装コスト1（低いほど重い）。
- **survey**: docs/45 §5・2026-06 調査。登録 2026-07-03。

### I-3 💡 SDF 縮小接線による sleeve 合成の置換（方法論枠）
- **出典**: Kozak, Nagel & Santosh (2020) "Shrinking the cross-section" JFE 135(2)＋
  本 repo docs/46 Stage1（H1 支持: SDF 目的が GKX 予測・単純合成を上回る +0.38 vs −0.23/−0.19）。
- **機構**: 実運用の sleeve 合成（現行 50/50 等加重）を KNS 縮小接線ポートフォリオに置換する。
  新しいαではなく**既存 sleeve の結合方法**の改善＝方法論候補。
- **元市場と主結果**: 米国 XS＋本 repo の JP 検証で H1 部分確認済み（docs/03 §6.21）。
- **JPデータ**: 既存 sleeve リターン系列のみ＝追加データ不要。
- **事前スクリーン**: αでなく配分手法＝judge_grid の枠に収まらない。検証するなら
  「固定 50/50 vs KNS 縮小」の事前登録比較（サブ期間・OOS）＋採用は人間ゲート（Phase 2 変更）。
- **スコア**: 機構4・新規性3・データ適合5・実装コスト4。
- **survey**: docs/46 H1 収穫。登録 2026-07-03。

### I-4 ⬆ 非線形TSMOM（S字カーブによる極端シグナルの減衰）
- **出典**: Moskowitz, Sabbatucci, Tamoni & Uhl (2025) "Nonlinear Time Series Momentum" SSRN 5933974
  （著者ページで確認: sites.google.com/site/riccardosabbatucci/Research・SSRN本体は403）。
- **機構**: トレンド→期待リターンの写像はS字＝極端な読みは増幅でなく減衰すべき。固定関数形なら1-2パラメータ。
- **元市場**: マルチアセット先物。非線形 SR 0.83-0.84 vs 線形 0.70（OOS・株式DD局面に利得集中）。
- **JPデータ**: 本番 TSMOM 11資産にそのまま重畳可（日次終値のみ）。
- **スクリーン**: momentum代理（本番スリーブとの増分が本丸）。関数形は事前固定（H-11）。
- **スコア**: 機構4・新規性3・データ適合5・実装コスト5。→ **trend_structure に昇格**（survey 2026-07-03）。

### I-5 ⬆ トレンド速度バーベル（中間ホライズンの冗長性）
- **出典**: Etienne, Ohana, Benhamou et al. (2025) "Revisiting the Structure of Trend Premia" arXiv:2510.23150（abstract 確認済）。
- **機構**: EMAラダーの中間帯（~125日）は隣接帯に張られ冗長＝短期＋長期のバーベルで足りる。
- **元市場**: 先物24銘柄 2013-25。等加重5ホライズン SR0.79→動的0.86・125日層除去で改善。
- **JPデータ**: 日次終値のみ＝11資産で可。債券/商品が無い分、彼らの分散効果は再現しない。
- **スクリーン**: 速い脚（20-60日）の回転がコスト死線（H-3）。I-6 と主張が正面対立→同一グリッドで裁く。
- **スコア**: 機構3・新規性3・データ適合3・実装コスト4。→ **trend_structure に昇格**。

### I-6 ⬆ 単一EMA最適性（チェリーピッキング回避のトレンド設計）
- **出典**: Valeyre (2025) "Breaking the Trend: How to Avoid Cherry-Picked Signals" arXiv:2504.10914（abstract 確認済）。
- **機構**: 一時間尺度の平均回帰ドリフト過程なら Sharpe vs シグナル速度の理論曲線が閉形式＝
  正しい速度の単一EMAで十分（R²≈0.98・最適半減期≈78営業日）。複雑な指標束は cherry-picking。
- **元市場**: 先物70銘柄 1990-2023・ピークSR≈1.24（リスクパリティ下）。
- **JPデータ**: 日次終値のみ。1パラメータ＝最小データ安全。**I-5 と正面対立**＝三つ巴比較が最安の検定。
- **スコア**: 機構4・新規性3・データ適合4・実装コスト4。→ **trend_structure に昇格**。

### I-7 💡 実時間ボラ管理の修正（条件付きSharpeスケーリング）
- **出典**: Xu (2024) "Improving Volatility-Managed Portfolios in Real Time" CFR forthcoming
  （cfr.ivo-welch.org PDF 確認済・SSRN 4778937）。
- **機構**: 素の1/σ²（Moreira-Muir）は実時間で失敗＝条件付き平均と切片を入れ「条件付きSharpe」でスケール。
- **元市場**: 米197ファクター。148/197 で Sharpe 改善・コスト後も生存（素は死ぬ）。
- **スクリーン**: **F5（de-risk）prior が強い**（本 repo は vol ゲート系を複数回棄却・H-5）。買い持ち超過を
  副次基準にした一発検証は可能だが優先度中。I-20（ポートフォリオ水準）と統合検討が先。
- **スコア**: 機構3・新規性2・データ適合4・実装コスト4。

### I-8 💡 モメンタムの日次重み学習（情報は決算日・ジャンプ日に疎在）
- **出典**: Beckmeyer & Wiedemann (2025) "All Days Are Not Created Equal" JBF 181, 107565（RePEc 確認済）。
- **機構**: 12-1 の等加重形成期間は誤り＝決算日・市場ジャンプ日・大幅特異リターン日が予測情報の大半
  （~30日で重みの50%超）。再加重モメンタムが古典を包含。
- **元市場**: 米XS・直近も含め既存モメンタムを subsume。
- **JPデータ**: 日次パネル＋PIT決算日で「日種別4パラメータ」制約版が組める（NN不要・H-11 準拠）。
- **スクリーン**: 日本はモメンタム弱（F3リスク）＋決算日成分は PEAD 隣接（要 dedupe 設計）。ただし
  「再加重こそが復活させる」という主張自体が日本での検定価値＝別シグナル。次候補圏。
- **スコア**: 機構4・新規性4・データ適合4・実装コスト3。

### I-9 ⬆ TSE資本効率イニシアチブの再価格付け（低PBR×高ROE・開示イベント）
- **出典**: D'Ercole, Wagner & Yamada (2025/2026) "Reputational Shocks and Capital Market Responses:
  The TSE Capital Efficiency Initiative" CEPR DP19971 / J. Corporate Finance 99, 103009（RePEc 確認済）。
- **機構**: 2023-24 の東証要請＋2024-01 対応一覧表が salience shock＝**低PBR×高ROE**の慢性割安銘柄が
  再価格付け。PBR=1 の不連続でなく割安度に滑らか・投資家側の repricing（利益改善なし）。
- **JPデータ**: TDnet タイトル/タグで「資本コスト…意識した経営」開示が特定可＋PIT PBR/ROE。
- **スクリーン**: F1（value共変）が本丸リスク＝**開示イベントのタイミングが静的 value ティルトを超える
  増分**だけが検定対象（value 残差化必須）。→ 既存 BACKLOG **governance_event_value の出典アンカー**として
  設計に反映（新規項目ではなく既存強化）。
- **スコア**: 機構4・新規性3・データ適合4・実装コスト3。

### I-10 🗑 株主還元アナウンスメント・ドリフト（割安条件付き）
- **出典**: Uchiyama & Takahashi (2022/23) "Post-Payout-Announcement Drift in Japan" SSRN 4619875。
- **棄却理由**: **F7 既試redux**＝scope `shareholder_return`（docs/03 §6.17・DSR0.24 FAIL・配当ドリフトは
  PBR改革後に反転）と同機構。差分の「割安条件付け」は F1（value共変）かつ governance_event_value の
  設計論点に吸収。2024年の記録的自社株買い（¥16.8兆）という母数拡大は §6.17 再訪の材料にはなるが
  単独の新仮説ではない。

### I-11 ⬆ ジャンプテールβのクロスセクション（N225 OTMプット→個別株テール感応度）
- **出典**: Alexiou & Rompolis (2024) "Jump Tail Risk Exposure and the Cross-Section of Stock Returns"
  J. Empirical Finance 79（research.ed.ac.uk で abstract 確認済: high-low **−9.95%/年**・下側テール主導）。
- **機構**: 指数オプションの deep-OTM プットからモデルフリーの左テール尺度→個別株リターンの
  テールβ推定→高βはテールヘッジ需要で買われ将来劣後（負のプレミアム）。
- **JPデータ**: `options_225`（2016-06〜・日次2,620ファイル・IV/Strike/UnderPx）＋全銘柄日次＝完結。
- **スクリーン**: 既試 vol_premium_n225（VRP売り）・option_regime_topix（ゲート）とは**別物**（XS 利用は初）。
  低ボラ/BAB 代理リスク（low_risk_anomaly と要直交化・H-1 類推）。OTM 板の疎さ→EVT 推定の安定性が
  実装リスク。→ **jump_tail_beta_xs として昇格**。
- **スコア**: 機構4・新規性5・データ適合4・実装コスト2。

### I-12 ⬆ スパースジャンプモデルによるファクター配分（少状態レジーム）
- **出典**: Shu & Mulvey (2024) "Dynamic Factor Allocation Leveraging Regime-Switching Signals"
  arXiv:2410.14841（確認済）。
- **機構**: 各ファクターの active return に2状態スパースジャンプモデル（ジャンプペナルティで
  偽スイッチ抑制・特徴選択内蔵）→ Black-Litterman の view に。IR ~0.05→0.4-0.5（米・OOS）。
- **JPデータ**: 既存のテスト済みファクターリターン＋マクロ系列で可。単一銘柄日足HMM（打ち切り）とは
  別物＝クロスセクショナル検証の再開条件に合致。
- **スクリーン**: H-11（~120ヶ月でレジーム2回＝1ブレイクへの過適合）・F5（momentum退出専用ゲート化）
  リスク。ジャンプペナルティは事前固定。→ 既存 BACKLOG **sjm_per_factor_regime の出典アンカー**。
- **スコア**: 機構3・新規性3・データ適合4・実装コスト3。

### I-13 💡 歪度分散（クロスセクション skewness dispersion）による市場予測
- **出典**: Babiak, Barunik & Kurka (2026) "Skewness Dispersion and Stock Market Returns"
  arXiv:2604.07870（確認済）。
- **機構**: 個別株実現歪度の**分散**（水準でない）が市場リターンを負に予測（信念の異質性/宝くじ需要の集中）。
- **スクリーン**: タイミング系＝F5 リスク（H-5: 買い持ち超過を副次基準必須）。論文は日中データの
  実現歪度の可能性＝日次代替は要事前登録の逸脱。XS 寄与度ランク版なら F5 を回避できる可能性。
- **スコア**: 機構3・新規性3・データ適合3・実装コスト3。

### I-14 💡 階層ベイズ縮小による戦略動物園の評価（方法論枠・判定意味論の変更）
- **出典**: Jensen, Kelly & Pedersen (2023) "Is There a Replication Crisis in Finance?" JF 78(5)
  （NBER w28432/Wiley 確認済・コード公開 github.com/bkelly-lab/ReplicationCrisis）。
- **機構**: 戦略αをクラスタ→全体の階層事前分布から縮小推定＝**相関する兄弟戦略は罰でなく証拠**。
  多重検定の論理を反転（多数観測が証拠を強める）。
- **platform適用**: 痛点(a) SR~0.4 単独因子の認定不能・(b) scope V[SR] 膨張（margin_alert_event で実体験）
  への最有力回答。永続レジストリ＝この推定の理想的入力。**判定基準の変更＝人間サインオフ必須**
  → まず PBO 同様の表示専用で併走が筋。
- **スコア**: 機構5・新規性4・データ適合5・実装コスト2。

### I-15 💡 e-backtesting（anytime-valid な運用監視・方法論枠）
- **出典**: Wang, Wang & Ziegel (2022-26) "E-backtesting" arXiv:2209.00991 / Management Science
  DOI 10.1287/mnsc.2023.01659（確認済）。
- **機構**: e-プロセス（非負スーパーマルチンゲール）で「戦略は主張通りか」の証拠を累積＝**連続的に
  覗いても・任意時点で止めても第一種過誤が制御**される。ホールドアウトの一回性(c)を解消。
- **platform適用**: Phase 2 稼働スリーブと C1 フォワード監視の**事前登録可能なキルスイッチ**
  （e値が1/αを超えたら統計的に認定された減衰）。判定意味論は不変（監視層の追加のみ）。
- **スコア**: 機構5・新規性4・データ適合5・実装コスト3。

### I-16 💡 IS/OOS Sharpe の厳密有限標本分布（方法論枠・drop-in）
- **出典**: Kan, Wang & Zheng (2024) "In-sample and out-of-sample Sharpe ratios of multi-factor
  asset pricing models" JFE 155, 103837（RePEc/著者PDF 確認済・複製コードあり）。
- **機構**: 推定ウェイトの戦略の IS/OOS Sharpe の**厳密分布**＝PSR の漸近正規近似を T=120 で置換可能。
  「このスリーブを足すと接線ポートフォリオの期待OOS SRが（推定リスク込みで）上がるか」という
  **スパニング・ゲート**の基礎。
- **platform適用**: 痛点(a)。minTRL≈20年の再計算・単独認定に代わる「結合貢献」判定の候補。
- **スコア**: 機構4・新規性4・データ適合5・実装コスト4。

### I-17 💡 複製率（replication ratio）の閉形式事前予測（方法論枠・drop-in）
- **出典**: Jacquier, Muhle-Karbe & Mulligan (2025) "In-Sample and Out-of-Sample Sharpe Ratios for
  Linear Predictive Models" arXiv:2501.03938（確認済）。
- **機構**: 線形予測モデル戦略の「IS Sharpe の何割が OOS で残るか」を #シグナル・#資産・シグナル強度・
  訓練窓長の閉形式で事前予測＝DSR（選択バイアス）と相補の**構造的過学習ハーカット**。
- **platform適用**: 痛点(a)(c)。予測 vs 実現の乖離が「過学習でなくレジーム断絶」の診断にもなる(d)。
  表示専用で導入可。
- **スコア**: 機構4・新規性3・データ適合5・実装コスト5。

### I-18 💡 合成環境での検証器の検証＋Bagged/Adaptive CPCV（方法論枠）
- **出典**: Arian, Norouzi Mobarekeh & Seco (2024) "Backtest overfitting in the machine learning era"
  Knowledge-Based Systems（SSRN 4686376/OUCI 確認済）。
- **機構**: 既知の真値（注入α・レジームシフト）を持つ合成市場で検証スキーム自体の判別力を測定。
  CPCV が K-fold/walk-forward を支配（本 repo 構成の独立追認）＋Bagged/Adaptive CPCV 拡張。
- **platform適用**: 痛点(c)(d)＝繰り返し可能な擬似OOS・レジームブレイク・ストレステスト。
  判定意味論不変（メタ検証モジュール）。
- **スコア**: 機構3・新規性3・データ適合4・実装コスト2。

### I-19 💡 1/N と平均分散の閉形式最適結合（方法論枠・I-3 と共同評価）
- **出典**: Lassance, Vanderveken & Vrins (2024) "On the Combination of Naive and Mean-Variance
  Portfolio Strategies" JBES 42(3)（RePEc/T&F 確認済）。
- **機構**: 標本接線と 1/N の最適結合＝凸制約を外し、その推定誤差を閉形式縮小で処理。T=120・N=2-4
  はまさに対象領域。現行 50/50 は一端点・I-3（KNS接線）は他端点＝**この結合が両者の橋**。
- **スクリーン**: N=2 では EW にほぼ縮退の見込み（複雑性に見合わぬ可能性）・iid 仮定と歪んだ月次 L/S。
- **スコア**: 機構4・新規性3・データ適合4・実装コスト4。

### I-20 💡 ボラ管理はスリーブ水準でなくポートフォリオ水準で（方法論枠）
- **出典**: DeMiguel, Martín-Utrera & Uppal (2024) "A Multifactor Perspective on Volatility-Managed
  Portfolios" JF 79(6)（Semantic Scholar/LBS OA PDF 確認済）。
- **機構**: ファクター毎の 1/σ² は OOS・コスト後に失敗（既知）。**結合ポートフォリオを市場ボラで条件付け**
  （リスク価格はボラ上昇で低下）だと純額でも生存。
- **platform適用**: 「ボラターゲットはどの水準で当てるか」に決着＝2スリーブ合成の上に N225 マイクロ先物
  1本で実装。I-7 より優先。採用は人間ゲート（Phase 2 変更）。
- **スコア**: 機構4・新規性3・データ適合4・実装コスト4。

### I-21 💡 リバランス・タイミング運とトランチングの損益分岐（方法論枠・最安の即効枠）
- **出典**: Zarattini & Pagani (2025) "The Tranching Dilemma" SSRN 5747964
  （concretumgroup.com/papers で PDF 確認済）。
- **機構**: 同一戦略でもリバランス日で CAGR 最大~350bps 差（RTL）。トランチングは RTL を減らすが
  取引回数増＝**AUM 依存の損益分岐**（小口は不利）。
- **platform適用**: ①旗艦スリーブを全リバランス日でリランして**自分の RTL バンドを測る**（単一バックテスト
  数値の解釈が変わる＝判定表示の改善・K外の診断）②現行 ¥100万ではトランチング不採用・¥1億で再訪。
- **スコア**: 機構3・新規性3・データ適合5・実装コスト5。

### I-22 💡 分布ロバスト対数最適（Wasserstein Kelly・コスト内包）
- **出典**: Hsieh & Yu (2024) "On Cost-Sensitive Distributionally Robust Log-Optimal Portfolio"
  arXiv:2410.23536（確認済）。
- **機構**: 経験分布の Wasserstein 球内の最悪ケースで対数成長を最大化＝フラクショナル Kelly の
  「割引率」を原理化。コストありだと安全資産側へ内生的にシフト。
- **スクリーン**: H-11 リスク（半径選択が新たな自由パラメータ化＝in-sample 誘惑）。より単純な
  Busseti-Ryu-Boyd (2016) の DD 制約 Kelly が先手（near-miss 参照）。優先度低。
- **スコア**: 機構3・新規性3・データ適合3・実装コスト2。

### I-23 💡 ターゲットボラのノートレード・バンド（方法論枠・優先度低）
- **出典**: Bai, Pachamanova, Steblovskaya & Wallbaum (2025) "Target volatility strategies: optimal
  rebalancing boundary…" FMPM, DOI 10.1007/s11408-025-00486-5（Springer 確認済）。
- **機構**: ターゲットボラ配分の周りにバンド φ＝逸脱時のみ戻す（1パラメータ・コスト削減）。
- **スクリーン**: 利得源は日次チャーンの抑制＝**月次リバランス＋マイクロ先物1枚粒度ではバンドが
  ほぼ発火しない**見込み（丸めが事実上のバンド）。I-20 採用時の付属論点として保持。
- **スコア**: 機構3・新規性2・データ適合4・実装コスト4。

（以降、/research-scout が追記）
