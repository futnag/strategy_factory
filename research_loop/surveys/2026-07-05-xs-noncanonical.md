# survey 2026-07-05-xs-noncanonical — 非カノニカルな日本株 XS ファクター（2024-2026 文献）

**テーマ（人間指定）**: 2024-2026 の最新文献から、tired-canonical でも既試でもない**クロスセクショナル**
日本株ファクター仮説。除外＝value/momentum/STR/低ボラ・BAB・IVOL/size/quality/PEAD/asset growth/
DSO/net-issuance/52週高値/信用オーバーハング/空売り残/指数入替/アクティビスト/TOB arb/自社株買い・
配当の発表 or 執行/予想バイアス/開示タイミング/GKX ML/共和分ペア/cheap disclosure-text。value or
momentum の機械的代理も除外。regime フィルタ＝2023-26 資本規律で value のみ耐久・US 符号反転・
一回性 2024 ショック依存を避ける。ローカル完結・小口が月次〜日次で執行可能なもの。

**起動理由**: 同日 morning の 2026-07-05-regime-durable が (a)方法論改良/(b)資本規律機構/(c)供給フロー を
掘り尽くした（I-50〜I-56・昇格2）。本 scout は**別コーナー**＝behavioral/clientele/higher-moment/日本発
2024-26 の未踏軸を fan-out。IDEAS は既に 💡 多数＝新規収穫は「除外リストと既存 features の両方を回避する
未踏機構」に限定。

## 手法
4並列 general-purpose サブエージェント（WebSearch/WebFetch）。各に I-1〜I-56 全 dedupe・§2 除外・
**既存 Gold features 一覧**（max_ret/ret_skew/ivol/beta/dimson_beta/seasonality/vpin 等＝MAX/歪度/IVOL は
実質既収載＝提案不可）・登録済 60 scope・スクリーニング prior（H-1/H-3/H-4/H-12/H-15/H-19/H-21・F1〜F9）を付与。
各エージェントに「WebFetch で実在＋abstract 一致を確認したものだけ返す・捏造は破棄」を義務化。返却後に
**親側で load-bearing 引用を再 WebFetch 検証**（scout Step 3）。
- Agent 1: overnight/intraday リターン分解の XS（tug-of-war・clientele）
- Agent 2: return seasonality（Heston-Sadka）＋ salience theory XS
- Agent 3: 符号付き第二モーメント（realized semibeta・downside/upside beta・coskewness）
- Agent 4: 日本発 2024-26 の novel XS 機構 ＋ 配当支払月の予測可能価格圧力（clientele フロー）

## 引用実在性チェック（Step 3・親側で再 WebFetch 検証したもの）
- ✓ Lou, Polk & Skouras (2019) JFE 134(1):192-213 — RePEc で確認（*"profits...earned entirely overnight
  ...or entirely intraday, typically with profits of opposite signs"*）。
- ✓ Bogousslavsky (2021) JFE 141(1):172-194 — 確認（margin/貸株コストで overnight component 持続）。
- ✓ Ho, Hsiao, Lo & Yang (2023) PBFJ 82, 102151 — 確認（台湾 IMOM(+)/OMOM(−)・12ヶ月持続）。
- ✓ Iwanaga & Hirose (2026) PBFJ 96(C), 103063 — 確認（*"stocks with high illusion momentum tend to have
  higher future returns"*・**日米両市場**・複利 vs 単純和の知覚ギャップ）。
- ✓ Hartzmark & Solomon (2025) AER 115(9):3171-3213 — 確認（配当支払の buying pressure が市場リターン予測）。
- （Agent 側 WebFetch 確認・親は abstract 転記で受領）: Chen & Kawaguchi (2018) IJEF 10(1) J-REIT tug-of-war・
  Bollerslev-Patton-Quaedvlieg (2022) JFE 144(1) realized semibetas・Li-Li-Su (2024) Applied Economics 57(26)
  **豪州で符号反転**・Atilgan et al. (2018) JPM 44(7) 国際 null・Ta (2015) RBES 3(2) **日本配当月 null**・
  Cosemans-Frehen (2021) JFE 140(2) salience・Cakici-Zaremba (2022) JFE 146(2) salience=STR代理（49ヶ国）。

## 収穫（IDEAS I-57〜I-61）
- **I-57 ⬆ overnight_intraday_tugofwar**（Agent 1・**昇格**）— 寄り=個人/海外 vs 引け=国内機関の clientele
  tug-of-war。overnight/intraday 成分を日足OHLCで分解＝features/registry に皆無の新軸。日本 REIT で機構実在
  証拠あり（Chen-Kawaguchi）。behavioral＝value 非変装。**最有力**（新規性5）。
- **I-58 ⬆ illusion_momentum**（Agent 4・**昇格**）— **2026 PBFJ・日本主標本**＝複利と単純和の知覚ギャップ。
  日次リターンのみ・実装最軽量。**H-15 が例外的に事前充足**（日本で方向確立＝scout で稀）。
- I-59 💡 realized_semibeta（Agent 3）— 符号付き4分解ダウンサイド共変。**符号反転 prior が極めて強い**
  （豪州反転・国際 null・本 repo で BAB/tailβ 既に反転）＝着手なら反転符号を expect して事前登録・優先度低。
- I-60 💡 dividend_month_price_pressure（Agent 4）— 配当支払月の予測フロー。**日本 null 既存(Ta 2015)**＋
  6月/12月集中で breadth 薄＝NISA レジーム条件付き再訪のみ・優先度低。
- I-61 🗑 salience / 銘柄季節性 / ファクター季節性の統合棄却（STR代理・既存 feature 重複・F5/H-21 墓場）。

## near-miss（拾わなかった理由）
- **salience theory**（Cosemans-Frehen 2021）: Cakici-Zaremba (2022・49ヶ国) が「主に短期リバーサル＋
  マイクロキャップ」と実証＝除外（STR）に吸収。→ I-61。
- **Heston-Sadka 銘柄季節性**: 既存 feature `seasonality` と重複＋日本履歴~10年で検定力不足。→ I-61。
- **ファクター季節性オーバーレイ**（Keloharju et al. 2021 / Mercik）: スリーブタイミング＝F5/H-21 墓場。→ I-61。
- **BPQ (2025) Granular Betas / Nevrla (2024) Asymmetric Risks**: 関数回帰・IPCA 11測度＝小口月次で実装不能・
  US-only・α が momentum に吸収。context のみ。
- **Zirk-Sadowski & Hryckiewicz (2025) FRL / Khan et al. (2025) RQFA / 中国 day-night**: **分足ティック必須**＝
  ローカル日足で計算不能（時間帯 intraday momentum・30秒〜60分）。
- **"Foreign Signal Radar" (Jiao 2025, arXiv 2504.07855)**: 日本は source market で**US 株を予測**＝XS の
  対象市場が違う。
- **TSE 資本効率 repricing / BOJ ETF フロー / R&D アノマリー**: value 変装＋2024窓依存 / 一回性(2024-03 終了) /
  quality 隣接＋pre-2024＝除外・既 backlog（governance_event_value・nikkei225_cap_flow・intangible B/M）に吸収。
- **"Does Overnight News Explain Overnight Returns" (arXiv 2507.04481)**: ニュース本文必須（F9）。
- **semivariance premium（option-implied − realized）**: 個別株オプション必須（N225 指数のみ＝不可）。

## 所感・フォローアップ
- **最良の2軸は behavioral/clientele（overnight/intraday）と日本発 2026（illusion momentum）**＝どちらも
  ローカル完結・方向固定可・value/momentum 非変装。特に **illusion_momentum は日本主標本＝H-15 を事前充足**
  （本 repo の候補の大半は H-15＝US 符号反転で死ぬ・これは日本で方向が既に立証）＝**最も安い検定**（1式・
  日次リターンのみ・K 効率高）。overnight は新規性最大だが H-15（寄り clientele 差）＋anchor 2019（H-19）で
  両方向前提の K 予算。
- **higher-moment リスクプレミアム軸（semibeta）は fan-out の主目的が「なぜ効かないか」の事前証拠収集に転化**＝
  Agent 3 が豪州反転＋国際 null を持ち帰り、**H-15 の base-rate を定量的に補強**（BAB/tailβ に続く3例目の
  ex-US 符号反転証拠）。これ自体が LESSONS 資産（着手前に負符号を予期できる）。
- **⚠ IDEAS が 61 件＝運用ルール 50 超**。次回 scout/retro で 🗑（I-10/24/30/31/39/42/49/56/61）・🧪 判定済み
  （I-4/5/6/9/11/12/29/32/33/37/43/51）をアーカイブ節へ移す保守が必要（本文は削らない）。
- 昇格2件で BACKLOG ⬜ が nikkei225_cap_flow に加え3件。次 /operator は規則5（red-team→cycle）。
  **推奨着手順＝illusion_momentum（最安・H-15 充足・日次のみ）→ overnight_intraday（新規性最大）**。
  ただし優先規則は operator 判断。
