# red-team: governance_event_value（2026-07-05）

**対象**: BACKLOG ⬜ `governance_event_value`（東証PBR改革開示イベント条件付き value・出典 I-9 =
D'Ercole, Wagner & Yamada 2026 CEPR DP19971 / JCF 99）。本日 /data-acquire D-2 で
`data/tse_capital_disclosure/`（13月・2025-05→2026-05）が解禁され ⏸→⬜。
**判定 = revise（重大・要再レビュー）**。判定器（judge_grid）を今の13月データに掛けるのは
**論文の機構と別の窓・別の仮説を検定することになり K を捨てる**。下記の再枠付けと K=0 cheap-kill を先に。

---

## §A 中心的欠陥 — 「検定したい窓」と「手元にある窓」の不一致（最重要）

- **論文（I-9）の機構は 2023-24 の要請＋2024-01 対応一覧表の salience shock** で低PBR×高ROE が
  **一度 repricing** される、という**イベント時点の水準シフト**。
- **手元の D-2 パネルは 2025-05→2026-05 の13月**（list.xlsx は ~13月ローリング）。
  **2024 の shock は窓外**。Prime は 2025-05 時点で既に ~90% 開示＝**左側打切り**（shock 済み・変動なし）。
- 窓内に残るのは2種類だけで、**どちらも論文の機構ではない**:
  1. **開示済/検討中 の静的ステータス**（2025 時点）＝緩慢な firm 特性。低PBR×高ROE に条件付ければ
     **value×quality そのもの**＝既試・除外（BACKLOG §2 手法/XS 系・F1/F7 redux）。
  2. **窓内の新規開示 ~250件（主に Standard・2025-06/07 偏在）**＝実イベントだが、salience が抜けた
     後の遅れた compliance（H-9/F6 減衰・H-17 即時織り込み）＝薄い事前分布＋Standard 小型（H-3/H-12/F8亜型）。
- ⇒ **今の窓で judge_grid を回すと、論文の主張（2024 shock repricing）でなく「2025 の遅れた
  Standard 開示ドリフト」または「静的 value×quality」を検定する**。前者は F6/H-17 で死に、後者は F1 既試。

## §B H-1〜H-22 機械照合（全数）

| H | 判定 | 根拠（条文と設計の突合） |
|---|---|---|
| H-1 momentum 代理 | △軽微 | 新規開示 firm は PBR テーマで直近上昇済みの可能性＝forward が mom 代理化しうる。統制に mom(12-1) を要 |
| H-2 value/PBR改革共変 | **●該当・本丸** | 仮説そのものが PBR改革レジームの value 復権＝**F1 の中心**。value 残差化が設計必須（BACKLOG §独立性で自認済）|
| H-3 コスト床 | **●該当** | 窓内新規は Standard 小型偏在＝月次15bps でも純化で消えうる（buyback は大型でも消えた前例）|
| H-4 minTRL | **●該当・致命** | 13月・in-regime＝認定構造的に不能（buyback と同じ）。標準 DSR は表示専用にせざるを得ない |
| H-5 de-risk | 非該当 | ロング傾斜のイベント増分＝de-risk 型でない |
| H-6 独立性≠エッジ | ●該当 | value 残差化で独立にしても「独立なだけの無エッジ」になりうる（H-6 の罠）|
| H-7 ショート脚衝突 | △ | value ヘッジ/ユニバースショートの構造衝突は限定的（ロング傾斜）|
| H-8 データ制約 | **●該当** | **論文機構の窓（2024）がローカルに無い**＝§A。faithful 検定は D-7 backfill 必須 |
| H-9 古データ減衰 | **●該当** | shock は 2024＝2025 の遅れた開示は**減衰後**（F6）。窓が減衰域に丸ごと入っている |
| H-10 イベント軸の構造優位 | △ | 「開示イベント」を軸にするのは妥当だが、salience が抜けた後のイベントは優位が消える |
| H-11 モデル複雑度 | 非該当 | 単純ティルト＝複雑度問題は無 |
| H-12 SR上限 breadth 概算 | **●該当・要 Stage-0** | 窓内新規 ~20-40件/月（Standard）×小型特異ボラ＝**H-12 で SR 上限を K=0 概算せよ**（margin_alert 型の F8亜型リスク）|
| H-13 規制/開示≠回復 | **●該当** | 「開示した」≠「実際に資本還元する」（talk is cheap）。開示ステータスは行動の代理として弱い |
| H-14 改良は standalone DSR 不可 | ●該当 | 「value の**増分**」＝既存 value スリーブの条件付け改良＝H-14 で proposals 枠になりうる（認定でなく）|
| H-15 米符号反転 | 非該当 | 日本主標本の論文＝反転リスク無 |
| H-16 Δ＝同窓価格代理 | **●該当** | 開示月と同月リターンは同時（repricing が上昇の原因）→ **forward（翌月以降）必須**。当月不使用 |
| H-17 日本開示は2024+ジャンプ即時織込 | **●該当・致命** | 本条は「post-disclosure **drift** をロングする」前提を直接否定。論文も**イベント時点の水準シフト**であって
  tradeable な後日ドリフトでない。**drift 型戦略は H-17＋論文の双方と整合しない** |
| H-18 統制/placebo | **●該当** | value 残差化（統制）は明記だが、**開示タイミングの placebo（同数・ランダム月の basket）＋
  turnover 中立（H-22）**がグリッド設計に未記載＝要追加 |
| H-19 古標本は方向のみ | ●該当 | 大きさ（DP19971 の CAR）を JP tradeable の期待値に流用するな＝方向のみ |
| H-20 指数フロー執行不能側 | 非該当 | 指数フロー案件でない |
| H-21 regime 切替 placebo 同値 | 非該当 | 状態切替オーバーレイでない |
| H-22 イベント basket の size/liquidity 交絡 | **●該当・本日学習** | **開示 firm は size に偏る**（Prime=大型飽和・Standard 新規=小型）＝raw basket は size ティルトを拾う。
  buyback で実証した通り**turnover 中立セル必須**（プラセボでは拾えない）|

## §C F-n 予想（死因の上位）
1. **F1（value/PBR改革共変・確率高）**: 静的版は value×quality 既試の別名。value 残差化しても H-6 で無エッジ化。
2. **F6（減衰・確率高）＋ H-17**: salience は 2024。2025 の遅れた開示は減衰後＋即時織り込みで forward drift ~0。
3. 支持: **F8亜型**（Standard 小型・少数イベント×特異ボラ×容量小）・**F9/H-8**（論文の窓がローカル欠＝
   faithful 検定不能）。§0 回避表に**「窓不一致（H-8/H-9）」と「H-17 drift 否定」が漏れる**と危険。

## §D PIT リーク狩り
- 開示ステータス: D-2 は各月末原本 snapshot＝PIT 安全（取得記録で確認済）。ただし **初出月＝開示イベント日**は
  月次粒度（日次でない）＝forward は「翌月以降」で設計すれば安全。`update_date` は「最終更新日」であって
  初回開示日でない＝**イベント日に使うな**（誤用すると遡及改訂リーク）。初出は「前月 blank/検討中→当月 開示済」で定義。
- 左側打切り: 2025-05 時点 開示済（Prime ほぼ全部）を「イベント」に含めるのは**shock を跨いだ遡及**＝
  実質 PIT 偽装（2024 の情報を 2025 の signal に混入）。**窓内 transition のみをイベント化**すること。
- ユニバース選択: Standard 新規開示 firm の生存/上場維持を forward で確認（`forward_returns_with_delisting`）。

## §E K 会計・統制セル（H-18）
- BACKLOG は value 残差化のみ明記。**統制セル不足**: (a) 開示タイミング placebo（ランダム月・同数）
  (b) turnover 十分位中立（H-22）(c) 「検討中」firm vs 「開示済」firm の対比（意思表明 vs 実開示の増分）。
- グリッドを組むなら ≤6・K≤8 だが、**§A の窓問題が未解決のまま K を投じるのは浪費**。まず K=0 で殺せる。

## §F 合格基準の well-formedness
- 「value 増分」は H-14 で **standalone 認定不可＝proposals 枠**になりうる旨を事前登録 §2.6 に固定せよ
  （trend_structure/H-14 前例）。in-regime 13月＝minTRL∞ を**事前受容**（buyback/sjm と同枠＝機構実在判定）。
- 事後救済の穴: 「窓内 tail が null なら 2024 backfill で再訪」は**別窓＝別サイクル**（新 prereg）であることを明記。

## §G AHM 7カテゴリ
- 動機: 論文実在・機構妥当だが**手元データが機構の窓を外す**＝動機と検定の乖離（要是正）。
- 多重検定: value×quality の分位違いを別仮説に数えない（§E 統制と混同しない）。
- データ: §A・§D。**最大の穴＝2024 窓欠落**。
- CV/モデル動学: in-regime のみ＝OOS 分離不能（F2 と同じ構造）。複雑度: 低。
- 文化: 「論文があるから効くはず」バイアス＝JP tradeable への流用は H-19 で方向のみ。

## §H より安価なキルチェック（K=0・実行前に殺す）
論文の窓（2024）を待たずとも、**手元の13月で "tail が死んでいるか" は K=0 で確定できる**（buyback 型 Stage-0 を
H-22 込みで最初から）:
1. **窓内 transition イベント**（前月非開示→当月開示済・主に Standard・~250件）を firm-month 化。
2. **forward t+1**（H-16）で、(a) 生 spread（対ユニバース）(b) **turnover 十分位中立**（H-22）
   (c) **value 残差化後**（PBR/簿価で中立化）の3系列。
3. **placebo**（同数・ランダム月 basket・H-18）と**「検討中→開示済」以外のダミー日**の null。
4. **ゲート**: value 残差化 × turnover 中立の forward 増分が (i) 正 (ii) placebo 外 でなければ **KILL**。
   （H-17/F6 の予測どおり ~0 になる公算大＝**documented F6 減衰 negative** を最安で得る。）
- これは judge_grid（K≥3）を回さずに tail を殺せる。**生き残った時のみ** D-7（2024 backfill）に投資して
  faithful 検定へ進む＝EV 最大の順序。

## §I 判定と修正案（revise・重大）

**判定 = revise（要再レビュー）**。理由: アイデア/論文は健全だが、**現データ窓が論文機構を外し、
今のまま judge_grid を回すと F1/F6/H-17 で確実に死ぬ別仮説を高コストで検定する**。

修正案（事前登録 prereg に反映すべき条項）:
1. **窓の分離を明示**: 「primary（論文機構＝2024 salience shock）は **D-7 backfill 取得後**の別サイクル。
   本サイクルは **in-window tail の cheap-kill（K=0 Stage-0）のみ**」と prereg §3 に固定。
2. **§H の K=0 Stage-0 を必須ゲート化**（value 残差化 × turnover 中立 × forward × placebo）。
   未達 KILL → **scope を ⏸ に戻し解除条件を「D-7（2024-01〜2025-04 backfill）取得」に更新**。
3. イベント定義を**窓内 transition のみ**に限定（左側打切り済みを除外＝PIT 偽装回避・§D）。
4. §0 回避表に **H-8/H-9（窓不一致・減衰）・H-17（drift 否定）・H-22（size 交絡）** を追加。
5. 合格基準に H-14（proposals 枠）・H-4（minTRL∞ 受容＝機構実在判定）を明記。

**次アクション推奨**: 本サイクル（operator 規則5 の cycle 段）では **§H の K=0 Stage-0 cheap-kill** を実施
（judge_grid は回さない）。tail が予想どおり null なら F6/H-17 negative を記録し ⏸（D-7 待ち）へ。
生存時のみ D-7 を data-acquire 優先度最上位に。
