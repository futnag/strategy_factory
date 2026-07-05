# survey 2026-07-06 — index オプション/IV サーフェス ＋ 時系列・マルチアセット

**scout**: /research-scout（引数なし・operator 規則6 発火＝BACKLOG ⬜1 に減少）。3テーマ fan-out の2テーマ分。
（テーマ「日本株 XS 新規ファクター」＝Agent A は別 survey `2026-07-05-xs-noncanonical.md`＋commit 41fac2e で独立記録済み。
本 survey は Agent B=オプション/IV・Agent C=時系列/マルチアセットの収穫を scout lead が検証・記録。）

## 手法
- Agent B/C（general-purpose・WebSearch/WebFetch）を並列起動。各に K=0・ローカルデータ制約・除外リスト
  （§2＋registry 60 scope＋既存 IDEAS 出典）・JP 小口執行可能性・失敗型事前予想を付与。
- **scout lead 独立検証**（guardrail 3）: 昇格候補＋load-bearing 引用を自身で WebSearch/WebFetch:
  - ✅ Driessen-Maenhout-Vilkov (2009) JF 64(3):1377-1406（corr risk・SSRN673425）— 確認。**「摩擦なし CRRA でのみ魅力」**を確認＝容量が本丸。
  - ✅ Pitkäjärvi-Suominen-Vaittinen (2020) JFE 136(1):63-85（cross-asset TSMOM・bond↔equity）— 確認。
  - ✅ Iwanaga-Hirose (2026) PBFJ 96 103063（illusion momentum・Agent A 分の再確認）— 確認。
- 独立確認できなかった引用は**記録せず**（下記 near-miss）。

## Agent B（オプション/IV）governing prior
**Andersen-Todorov-Ubukata (2021) J.Econometrics 222(1):344-363** = 日本自身のオプション由来テール/vol 指標は
**N225 リターン予測に有意でない**（効くのは US SPX テールのみ）。⇒ **N225-IV→N225-return 型は全て事前弱化**。
これが本テーマの最重要スクリーン。

## 収穫（記録）
| I-n | slug | 状態 | 要点 |
|---|---|---|---|
| I-62 | corr_risk_beta_xs | ⬆ **昇格** | XS β to implied-correlation innovation。ATU null を横断面ゆえ回避・F5でない・options_225 初活用。容量(DMV)/dispersion交絡が Stage-0 gate |
| I-63 | xasset_bond_equity_tsmom | 💡 | 債券↔株トレンド相互予測(+45%SR)。**債券TR系列がローカル欠**＝要データで保留 |
| I-64 | factor_momentum_jp | 💡 | ファクター・リターンの1M自己相関。JP全銘柄で完結だが**sjm(❌)直後＝H-21同族リスク**で保留 |
| I-65 | macro_econ_trend | 💡 | マクロ・ファンダメンタルのトレンド(AQR)。GDP予測改定欠で景気sleeve劣化 |
| I-66 | us_tail_to_n225 | 💡 | US テール→JP USD リターン/USDJPY(ATU 唯一のJP有意)。F5/FX carry 交絡が本丸 |
| I-67 | （統合🗑） | 🗑 | term-slope/signed-VRP（ATU null＋F5＋vol_premium近接）・xasset同月季節性（I-61重複） |

## near-miss（棄却・理由付き）
- **skewness_dispersion**（Babiak-Baruník-Kurka arXiv:2604.07870）: firm-level 実現スキューの XS 分散が市場を負予測。
  ローカル計算可だが **scout lead が arXiv を独立確認できず**＝guardrail 3 で**記録せず**（要再確認）。market-timing ゆえ F5 も。
- **OVX cross-asset TSMOM**（Fernandez-Perez et al 2022 JBF）: 原油**インプライド**vol 要＝ローカル日足のみで不可。
- **Network momentum**（Li-Ferreira arXiv:2501.07135）: ~50-100 先物設計＝~11 資産では graph 過学習。
- **Barbell trend**（Etienne et al arXiv:2510.23150）: trend 構成＝killed trend_structure/採用 TSMOM に近接。
- **Carry-crash / yen-unwind**（CEPR DP20745）: carry 資金調達不可＋「yen-unwind で de-risk」＝F5/H-21。
- **Currency basis-momentum**（Fan 2025 EFM）: forward/basis 不在で組めず。
- **Fournier (2024) JF**（conditional factor risk premia from option returns）: index-option リターンの横断面要＝重い。
- **Bondarenko-Bernard (2024) JFQA**: corr risk の model-free 厳密版だが個別 option 要＝I-62 の理論 anchor のみ。

## 昇格判断
- **1件昇格（corr_risk_beta_xs）**。理由: (a) options_225 の未活用データを活性化（戦略的）、(b) ATU-null を横断面ゆえ回避、
  (c) F5 でない XS dollar-neutral、(d) 2つの懸念（容量・dispersion 交絡）が**まさに Stage-0/red-team で安く殺せる**型。
- 他は保留: 債券TR欠（I-63）・sjm 直後の H-21 同族（I-64）・computability 劣化（I-65）・F5/FX（I-66）。
- Agent A（XS）が既に illusion_momentum・overnight_intraday_tugofwar を昇格済＝本 scout の総昇格 **3件**（XS2＋options1）。

## メモ
- **IDEAS が 67 件（>50）＝アーカイブ整理が過期**（🗑8・🧪12 の retro sweep を次回 /loop-retro or scout で）。
- 本 repo の地形再確認: N225-IV→N225 timing は ATU-null＋F5 で構造的に不利。オプション活用は**横断面リスク
  プレミアム**（corr/dispersion）に限る。マルチアセットは債券TR欠がボトルネック（→ data_ideas 候補）。
