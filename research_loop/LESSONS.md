# LESSONS — 失敗型タクソノミーと蓄積知見

**運用ルール**: append-only。§1 のタクソノミーは**追加のみ可**（新型は既存型で説明できない理由を
明記）。§4 に判定済みサイクル毎に1エントリ追記。本ファイルはループの知見蓄積層であり、
人間が定期的に `docs/03-research-findings.md`（正本）へ編入する。ループは docs/03 を編集しない。

---

## §1 失敗型タクソノミー（判定済み試行からの帰納）

| # | 型 | 内容 | 代表例（scope・最良DSR） |
|---|---|---|---|
| F1 | 直近regime共変 | 2023+ のみ効く＝資本規律(PBR改革)レジームで value と同根。独立源でない | asset_growth 0.78（vs value ρ̄+0.22） |
| F2 | OOS負・直近死 | IS(〜2023)で機能→2024+ OOS で負転。フォワードで死ぬ型 | conditional_sdf 0.52（IS+0.64→OOS−0.37）、macro_sector_rotation 0.56 |
| F3 | 符号逆 | 事前登録の仮説方向と逆に動く（実質別ファクターの裏面） | dso_quality 0.12（SR−0.24＝成長株売り）、BAB（低ベータ SR−0.7） |
| F4 | momentum代理 | momentum(12-1) の単調変換・高相関。新軸に見えて既試 | high_52w 0.67（ρ̄+0.38） |
| F5 | de-risk＝αでない | DDは減るが Sharpe 改善なし・買い持ち未満（露出調整に過ぎない） | option_regime_topix 0.83、flow_divergence_topix 0.56 |
| F6 | 古データ減衰 | 2016-19 で強く単調減衰→直近死（市場構造の変化） | margin_overhang 0.75（+1.15→−0.07） |
| F7 | 既試redux | 既判定ファクターの高相関な別名（ρ>0.8） | IVOL（low_vol と ρ0.82・low_risk_anomaly 0.15） |
| F8 | gross有・cost死・非スケール | gross は実在するがスプレッド/回転で net 死・容量極小 | limit_reversal 0.26（gross+1.08・損益分岐25bps・容量¥0.2百万） |
| F9 | データ制約 | 必要データがローカルに無く公正な検定が組めない（**Stage-0 で殺すべき型**） | bond_term_carry 0.25（分散キャリー不可） |

出典: docs/03 §6.20-6.21（9バッチ＋フロンティア2軸）・docs/48 §5・TODO.md（2026-06 消化記録）。

**F8 亜型（2026-07-03 追記・margin_alert_event）**: 「平均イベントドリフト実在・ポートフォリオ
変換で溶解」＝死因がスプレッドでなく**少数銘柄集中×高特異ボラ×容量ゼロ**。イベントスタディの
平均超過リターンが負に実在しても（−140〜−394bps/10日）、同時イベント数が日次1〜5では 1/E 等加重の
分散が効かず SR にならない。Stage-0 で SR≈(日次ドリフト/銘柄ボラ)×√実効breadth を概算せよ（→H-12）。

## §2 耐久 positive（数少ない生き残りと方法論的収穫）

- **value（B/M）**: 生存者バイアス補正後も生存・OOS(2024+) SR+0.49 ≥ IS+0.32・PSR0.87。
  ただし SR~0.36 は低く minTRL≈241ヶ月＝単独では認定不能（docs/03 §6.3・§7.1）。
- **value＋ロングティルトPEAD（旗艦）**: DSR0.91（scope局所）・OOS+0.62・maxDD−19.2%・容量≈¥6億。
  **留保: ロングティルト化は OOS を見た post-hoc 修正**＝真の検証はフォワード（§7.4）。
- **value↔PEAD レジーム切替**: 静的 switch DSR0.99・walk-forward 版 0.97（in-sample 懸念解消）（§6.10-11）。
- **TSMOM（11資産）×旗艦の2スリーブ**: 合成 OOS SR+1.05・maxDD−5.4%・DSR0.99（scope局所）＝
  現行 Phase 2 運用ポートフォリオ（§6.15-16）。
- **方法論的収穫**:
  - SDF 目的（価格付け）が予測ML（GKX）・単純合成を明確に上回る（+0.38 vs −0.23/−0.19）＝
    目的関数の選択は有効な自由度（docs/46 H1・§6.21）。sleeve 合成の SDF 縮小接線化は検討価値あり。
  - **監査・診断は K 消費ゼロで可能**（実装リスク cross-engine 突合・上流汚染 mask-first・
    factor mirage 監査＝§6.20(748)-6.27）。判定の前に診断を尽くすのは無料。
  - 失敗の「機構」まで診断できると経済的に修正できる（PEAD 脚分解→ロングティルト＝§8）。

## §3 メタ観察（最重要の地形認識）

ほぼ全ての候補が「**2016-23 で機能 → 2023+ で死/OOS負**」。2024-26 の資本規律（PBR改革）
レジームは系統的シグナルに敵対的で、value のみ耐久。「効いて見える新候補」の多くは
このレジームで value と同根（F1）。**platform（検証ファクトリ＋documented negative）が成果物**であり、
偽αを出荷しないことが勝ち（docs/03 §6.20 結論・TODO.md 2026-06-27 メタ観察）。

追加の運用教訓:
- 「立派に見える偽物」は実際に出る（size DSR0.82 が生存者バイアス由来だった＝§2.4）。
- 機械的な因子合成は通算複数回失敗。効くのは経済的に筋の通った合成だけ（§8）。
- kaiten（回転手法）系はレジストリ/DSR 規律の外＝設計根拠に使わない（メモリ規律）。

---

## §4 サイクル記録（append-only・新しいものを下に）

### 2026-07-05 buyback_execution_flow — ❌ FAIL（size/liquidity 交絡・F7亜型・best DSR0.80・K=3）※raw が本ループ2番目の接近だが交絡
- 仮説: EDINET「自己株券買付状況報告書」提出（＝実執行中）企業は t+1 forward 超過（scout I-51・Clarke2022 FRL・
  執行フロー persistence・docs/62）。red-team 2026-07-05 revise 全反映（データ源 TDnet→EDINET 訂正・submit アンカー・in-regime）。
- 結果: **Stage-0 生存**（生spread+0.38%/月・プラセボ p=0.018・turnover中立+0.12%）→ judge_grid（K=3）で **best bef_exec_ls
  DSR0.80 < 0.95**。決め手＝**turnover十分位中立 bef_exec_sizematch で SR−0.07・DSR0.21**＝raw の超過は執行フロー特異でなく
  **size/liquidity ティルト**（買付執行企業＝大型・高流動）。in-regime 13月＝minTRL 261月＝認定構造的に不能。
- 機構1行: **プラセボは通るが size中立で消える**＝「イベント」でなく「イベントをする企業の属性」が効いていた → **H-22 新設**。
  発表軸 shareholder_return❌（DSR0.24）に続き**執行軸も否定**＝自社株買い3軸目（発表/能力/執行）も棄却。
- 収穫: Stage-0 に turnover中立セルを必須化（H-22）＝プラセボ（H-18・イベント有無）と別軸の統制。executor 抽出
  （examples/research_buyback_execution_flow.py）は D-8 backfill 時に再利用可（ただし size中立αが無い限り再訪不可）。
- メタ: operator 規則5・第7起動（red-team）→本起動（反映+cycle）。**本ループ初の「Stage-0 生存→judge FAIL」**
  （sjm は Stage-0 kill）。raw DSR0.80 は trend_structure 0.92 に次ぐ接近だが、統制セルが偽陽性を解体した好例。
- 成果物: docs/62 §5・examples/research_buyback_execution_flow.py・data/reports/buyback_execution_flow.html。

### 2026-07-05 sjm_per_factor_regime — ❌ FAIL（Stage-0 KILL・F5＋H-18 placebo 同値・K=0）※初の Stage-0 K=0 kill
- 仮説: per-factor regime 切替（2状態 SJM・ジャンプペナルティ固定）が固定等ウエイト合成を上回る
  （scout I-12・Shu-Mulvey 2024・US 6因子 IR 0.05→0.4-0.5・docs/61）。red-team 2026-07-05 revise 全反映。
- 結果: **FF Japan 3因子・430ヶ月の best-case でも Stage-0 で KILL（K=0・judge_grid 未実行）**。
  switched SR **+0.306 < 固定 +0.671**（F5・de-risk 有害）・placebo(shuffle) **mean+0.309/p95+0.462 の内側**
  （H-18＝timing 情報ゼロ）・mom オーバーレイ +0.898 に劣後（H-1）。rho0.60・乖離88%＝動いて損なう型。
- 機構1行: **regime 切替は placebo 同値＝情報価値ゼロ**。遷移 median 5（推定可）＝**H-11 データ不足を反証**
  ＝機構そのものが JP 因子配分で無効 → **H-21 新設**。red-team の Stage-0 義務化で K=0 決着。
- 収穫: self-contained オンライン・ジャンプモデル（examples/research_sjm_per_factor_regime.py・因果 regime）は
  他の regime/timing 候補（実時間ボラ管理・日次重み学習等）の Stage-0 に再利用可。H-21 を prior 適用。
- メタ: operator 規則5・第4起動（red-team）→第5起動（反映+cycle）の連結が設計どおり完走。
  **Stage-0 K=0 kill は本ループ初**（judge_grid を回さず documented negative＝最安の FAIL＝red-team 内ループの本領）。
- 成果物: docs/61 §5・examples/research_sjm_per_factor_regime.py。

### 2026-07-04 topix_staged_flow — ❌ FAIL（F8亜型＋H-3＋容量死・DSR0.00・K=5）※初の自律 operator サイクル
- 仮説: TOPIX 段階削減（四半期末×10段階・2022-10〜2025-01）の実施日近傍リバーサル（明田2022・
  同日DiD・実効n=10・docs/60）。scout I-32・/data-acquire で解禁・red-team revise 全反映。
- 結果: **gross 機構は実在しプラセボもクリーンだが執行不能で FAIL**。gross リバーサル [1,11)
  **+75bps・8/10ステップ正**・時間プラセボ +16/−17bps（差+59bps）＝**削減特異で実施日に局在**。
  圧力[−1,+1]−62bps→リバーサル+97bps の形状も再現。だが net SR = 0bps **+0.53** → 30bps **−0.63**
  （借株300bps＋イベント30bps）・容量 **¥183万**＝小型ゆえ純化が消える。
- 機構1行: **指数フロー対象＝定義上その指数の最小 float 銘柄**＝借株高・容量極小で、機構が
  本物でも執行不能側に生まれる → **H-20 追加**。実効 n=10 は事前予告どおり認定不可（minTRL=∞）。
- 収穫: 機構実在の判定という主目的は達成。Stage2（2026-10 前向き）は**大型サブセット or 執行
  コスト実効低減の角度が無ければ EV 低**と確定（BACKLOG に反映）。
- メタ: 初の operator 規則5経由の自律サイクル。red-team（revise）→prereg→実装→判定が
  設計どおり連結。red-team ②（四半期末交絡）が gross プラセボ比較で明示的に解消された。
- 成果物: docs/60 §5・examples/research_topix_staged_flow.py・data/reports/topix_staged_flow.html。

### 2026-07-03 unexpected_forecast — ❌ FAIL（F6 減衰・DSR0.37・K=4）
- 仮説: 経営者予想の裁量部分 DF の過大評価と12ヶ月 unwind（scout I-43・JBFA2024 日本主標本・
  CARF F367 本文で仕様確認・docs/59）。
- 結果: **副次基準5つ全通過・主基準のみ未達**。グリッド序列は理論と完全一致
  （DF +0.11 ＞ FI −0.09 ＞ NDF統制 −0.34＝分解の識別が機能・H-18 の2度目の勝利）。
  分位単調性も方向どおり（Q5−Q1=+24bps/月）だが**経済的規模が1桁不足**（コスト0でも SR+0.13）。
- 機構1行: 2003-09 標本の効果は 2020+ で方向のみ残存＝**公表後裁定減衰（McLean-Pontiff 型）**
  → H-19 追加（標本年代で EV をディスカウント）。WF 学習のラグ消費（実効標本 2020+）も H-4 実装注意。
- 収穫: I-44（達成率持続）は独立軸と確認（DF vs 前年誤差 ρ=+0.02）が効果規模から優先度低。
- 成果物: docs/59 §5・examples/research_unexpected_forecast.py・data/reports/unexpected_forecast.html。

### 2026-07-03 margin_deadline — ❌ FAIL（機構不発・DSR0.00・K=4）
- 仮説: 制度信用6ヶ月期日のコホート供給（scout I-37・学術未検証の白地・H-17 適合型・docs/58）。
- 結果: **プラセボ設計により「機構不在」を一意に識別**＝期日窓 +8bps/週 ＝ プラセボ窓 +8bps/週の
  完全一致（期日局在なし・イベント27,063件で検定力十分）。乗り換え・早期決済で期日供給が霧散。
  「6ヶ月期日売り」の実務 lore は週次データでは観測されない。
- 機構1行: 統制セル1つで FAIL の解釈が一意になる → H-18 追加（イベント/フロー系はプラセボ標準装備）。
  「実務 lore×学術未検証」は未発見エッジとは限らず「検証に耐えなかった lore」の事前分布も持つ。
- 成果物: docs/58 §5・examples/research_margin_deadline.py・data/reports/margin_deadline.html。

### 2026-07-03 activist_filing_drift — ❌ FAIL（F2/F6＋F8亜型・DSR0.08・K=4）
- 仮説: 大量保有報告の filer 異質性ドリフト（scout I-33・日本主標本で方向固定・docs/57）。
- 結果: **異質性は再現**（名簿filer +386bps/5日 vs 全filer −73bps/40日・統制超えΔSR+0.95）が、
  織り込みは**ジャンプ型**（5日で完結→以後フラット）＝T+1 後の残余が細くSRに変換されず。
  年次 2022+0.97→2025−1.23 の標本内急減衰（アクティビズム・ブームの crowding）。
- 機構1行: **開示イベントの高速織り込み（2024+）**＝イベント平均効果の実在と取引可能性は別物
  → H-17 追加（T+1 以降に残る構造的理由の事前説明を要求）。
- 収穫: filer 名簿・重要提案フラグは除外/リスク管理スクリーン部品として有効。
  edinetCode→secCode 対応表（`data/processed/edinet_code_map.parquet`）は汎用再利用可。
- 成果物: docs/57 §5・examples/research_activist_filing.py・data/reports/activist_filing_drift.html。

### 2026-07-03 delta_liquidity — ❌ FAIL（F3 符号逆＋F4 亜型・DSR0.00・K=4）
- 仮説: 流動性「変化」の翌月ドリフト（scout I-29・日本主標本 PBFJ2023 で成分別方向を事前固定・docs/56）。
- 結果: 両仮説とも分位単調性が逆（Q5−Q1: ST−21bps/LT−30bps）。**設計欠陥を事後発見**＝
  12M流動性変化は momentum(12-1) と ρ̄−0.46＝同窓の価格変化との同時性で実質アンチ・モメンタム。
  事前登録コントロールに同窓 momentum が欠落（1Mリバーサルのみ入れていた）。
  回転も構造的に高く（1.7〜3.2回/月）gross≈0 → net 即死。
- 機構1行: **Δ系シグナル＝同窓価格系の機械的代理＋高回転**という二重の構造問題 → H-16 追加。
  日本主標本の査読済み証拠でも当方の操作的定義では再現せず（定義乖離 or 標本期間差＝F6 の可能性併記）。
- 成果物: docs/56 §5・examples/research_delta_liquidity.py・data/reports/delta_liquidity.html。

### 2026-07-03 jump_tail_beta_xs — ❌ FAIL（F3 符号逆・DSR0.02・K=4）
- 仮説: N225 OTMプット由来テール尺度への感応度 XS（scout I-11・方向は米国 JEF2024 で事前固定・docs/55）。
- 結果: 全セル負（主セル SR−0.67）＝**符号が米国と逆**。ただし急落月は全勝（2020-03 +1.4%・
  2024-08 +2.4%）＝ソーティングは保険分類器として機能し、日本では保険料を払う側だった。
  残差化でρ̄≈0 を達成＝「独立だが符号逆」（dso_quality と同型・H-6 再々確認）。
- 機構1行: **日本のクロスセクションは米国のリスクプレミアム系発見の符号を反転させる**
  （BAB→本件テールβ→momentum弱の系譜）→ H-15 追加。
- 収穫: LT尺度＋二変量ローリングβの基盤（`n225_tail_measure` キャッシュ）は再利用可。
- 成果物: docs/55 §5・examples/research_jump_tail_beta.py・data/reports/jump_tail_beta_xs.html。

### 2026-07-03 trend_structure — ❌ FAIL（DSR0.92・K=6・全試行中の最接近）
- 仮説: トレンド設計の構造改善（単一EMA/バーベル/S字 vs 12-1 サイン・docs/54・scout 由来＝
  方向は他市場文献で事前固定）。
- 結果: H1（単一EMA）棄却（ΔSR−0.23）。H2（バーベル）冗長性のみ確認（±0）。**H3 飽和型 tanh は
  全リスク指標でベースライン支配**（SR+0.48 vs +0.43・回転0.19 vs 0.25・maxDD−9.6% vs −13.5%）
  だが DSR 0.92 で認定未達。OOS(2024+)は全セル +1.4〜1.7＝機構は健在。
- 機構1行: **改良研究の standalone DSR はベースラインに張り付く**（ρ0.86）＝劇的改善以外は
  構造的に認定不能（H-4 の認定不能領域の新しい現れ方）→ H-14 追加。
- 収穫: tanh 減衰＝本番 TSMOM の無料改善候補（実装1行・人間判断待ち・docs/54 §5）。
  スパニング判定（IDEAS I-16）導入後の再判定条件を docs/54 に記録。
- 成果物: docs/54 §5・examples/research_trend_structure.py・data/reports/trend_structure.html。

### 2026-07-03 margin_alert_event — ❌ FAIL（DSR0.01・K=6）
- 仮説: 増担保/日々公表の指定→負・解除→正のドリフト（docs/53）。
- 結果: 解除ロングは**符号逆（F3）**＝解除後も下落継続（−566bps/10日）。指定ショートは
  ドリフト実在だが**F8亜型**＝少数集中×特異ボラ×容量¥万単位で SR に変換されず。
  独立性は確認（ρ̄≈0〜0.1）＝また「独立だがエッジ無し」（H-6 再確認）。
- 機構1行: **信用規制イベント銘柄は指定・解除を問わず持続的に劣後**（投機の脱気は解除後も続く）。
  ロング簿除外スクリーンとしての残存価値のみ（対象少・影響軽微）。
- 教訓→ H-12（ポートフォリオSR上限の Stage-0 概算）・H-13（規制解除≠回復）を追加。
- 成果物: docs/53 §5・examples/research_margin_alert.py・data/reports/margin_alert_event.html。
