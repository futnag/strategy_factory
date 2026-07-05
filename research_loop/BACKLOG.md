# BACKLOG — 仮説キューと dedupe 台帳

**運用ルール**: `/research-cycle` は §1 の最優先 ⬜ を1つ取って1サイクルを回す（1サイクル1仮説）。
新候補は**仮説／経済的根拠／データ／新規性／独立性／注意**の構造で追記。着手時 ⬜→🔬、
判定で ✅/❌、保留は ⏸（解除条件を明記）。dedupe は §2（既判定）＋§3（Stage-0 棄却）＋
registry の scope 一覧（`loop_status.py`）の3つに対して行う。

**状態凡例**: ⬜ 未着手 / 🔬 検証中 / ✅ PASS / ❌ FAIL / ⏸ 保留

---

## §1 キュー（優先順）

### ⬜ nikkei225_cap_flow — 日経225 上限係数(PAF)キャッピングの大型株・確定インデックスフロー
- **仮説**: 日経225 のキャッピング（PAF 切下げ）で実施日に全パッシブが対象 **mega-cap** を機械的に大量売却＝
  一時的価格圧力→数日〜数週で**リバーサル**（対象買い/ヘッジ）。方向は price-pressure（KMM 2000/Tu 2012）で事前固定。
- **経済的根拠**: 情報でなく**カレンダー付き強制フロー**（基準日 7月/1月末→実施 10月/4月・閾値公知）。指値供給者への一時
  プレミアムの巻き戻し（H-17 適合）。**対象が指数の最大・最流動・借株容易な mega-cap＝H-20 の明記例外「大型サブセット」**
  ＝topix_staged_flow（小型 float・借株300bps・容量死❌）の**執行可能な鏡像**。この設計で H-20 が予告した唯一の抜け道を検定。
- **データ**: J-Quants 日足（完備）＋日経225 構成・**PAF 履歴**（ガイドブック係数表＝半公開・軽度 F9＝§1 で取得確認）。実施日はルール確定。
- **新規性**: registry/§2 に「キャッピング確定フロー」なし。既試 index_events_n225（入替ドリフト❌・2020+消滅）・
  topix_staged_flow（段階削減・小型❌）とは**対象(mega-cap)・機構(ウエイト切下げ)が別**（F7 正当化を prereg §0 に明記）。
- **独立性**: フロー駆動 LS＝value/momentum と構造的に別軸。
- **注意（Stage-0 必須）**: **最大リスク＝執行可能だが price-impact が mega-cap ADV に対し小＝F8 の逆(エッジ無)**
  （Tu 2012: large-cap 反応は小型より小）。→ Stage-0 で「想定フロー額 / 対象 ADV」と想定 impact を **H-12 型 K=0 概算**し、
  SR 上限が認定圏か先に判定。実効 n も薄い（年2回×少数銘柄＝H-4）＝機構実在判定が主目的になり得る。借株容易ゆえコスト床は topix より緩い。
- **出典**: Kaul, Mehrotra & Morck (2000) JF 55(2):893-912（ウエイト調整の自然実験・残存−13%→反転）／
  Tu (2012) IJBFR 6(4):59-71（Nikkei225 入替 price-pressure・確認済）＝[IDEAS I-50]。日経ガイドブック上限比率ルール（2022導入・閾値12→11→10%）。

### ⬜ illusion_momentum — イリュージョン・モメンタム（複利 vs 単純和の知覚ギャップ）
- **仮説**: ルックバック窓で**単純和 Σr が複利 ∏(1+r)−1 を過大表示する**（＝ボラ drag 大）銘柄は、投資家が
  単純和を真のリターンと誤認し過小反応する結果、翌月以降リターンが**高い**（high illusion momentum ロング）。
  方向は Iwanaga-Hirose (2026) が**日本主標本で確立**＝事前固定（他市場覗き見でない）。
- **経済的根拠**: 取引画面/明細に表示される見栄えの良い単純和への認知バイアス（cumulative sum を
  cumulative return と誤認）。標準ファクター・特性と独立と原論文が報告・bear 局面で強化・大型でも残存。
- **データ**: J-Quants 日次リターン（`adj_close` 由来）のみ＝完全ローカル・最軽量（1式・月次 XS）。
- **新規性**: registry 60 scope・features（`max_ret/ret_skew/ivol/mom_*/reversal_*`）に**複利-単純和ギャップ**は
  皆無。"momentum-related" と自称するが trend 継続でなく knowledge-gap 構造＝momentum(12-1) と別軸。
- **独立性**: ⚠ **最大リスク＝ギャップ≈½·Σr²＝realized variance の再符号化**（ivol/rvol への機械的近接）→
  **主セルで ivol/rvol/parkinson ＋ momentum(12-1)/reversal_1m へ直交化**し残差セルの生存を副次基準に。
- **注意（Stage-0）**: **H-15 が例外的に事前充足**（日本で方向確立＝符号反転リスクが低い＝本 repo で稀な優位）。
  ただし高ボラティルトが low_risk_anomaly（低ベータ SR−0.7）と符号衝突しないかを直交化で確認。**H-4**（単一 XS
  特性 SR~0.4 は minTRL 認定不能圏＝breadth で緩和・スパニング枠 I-16 導入後が理想）。回転は月次で 15bps 床側。
- **出典**: Iwanaga & Hirose (2026) "Illusion momentum and cross-sectional returns" PBFJ 96(C) 103063,
  DOI 10.1016/j.pacfin.2026.103063（WebFetch 確認済・日米両市場）＝[IDEAS I-58]。

### ⬜ overnight_intraday_tugofwar — オーバーナイト/日中リターン分解の clientele tug-of-war
- **仮説**: 銘柄別の**オーバーナイト成分（`adj_open_t/adj_close_{t-1}−1`）と日中成分（`adj_close_t/adj_open_t−1`）**を
  月次窓（≥1M）で累積し XS ランク→各成分は clientele 継続で持続。方向は LPS(2019)/台湾 Ho(2023) の
  IMOM(+)/OMOM(−) で事前固定（他市場証拠）。日本での符号は両方向前提で prereg（H-15）。
- **経済的根拠**: 寄り＝個人/海外、引け＝国内機関の限界主体差＝ある銘柄を寄りで買い続ける clientele が
  オーバーナイト成分の継続を、逆側が日中で押し戻す cross-period reversal を生む（情報でなく需給 clientele）。
  日本 J-REIT で foreign/個人 vs 国内機関の tug-of-war が実在確認済（Chen-Kawaguchi 2018）。
- **データ**: J-Quants `adj_open/adj_close`（全銘柄2016+）のみ＝完全ローカル。粗い clientele 条件付けに
  `investor_types`（市場集計・海外/個人）を併用可。
- **新規性**: overnight/intraday 分解は features/registry に**皆無**（close-to-close の mom/reversal/seasonality のみ）。
  index_events 系・flow 系とも別軸＝behavioral clientele。
- **独立性**: ⚠ **F4/短期リバーサル代理**（形成窓を~1日にすると intraday=STR・overnight=1日 overreaction に縮退）→
  **形成窓 ≥1ヶ月固定・momentum(12-1)/reversal_1m へのスパニングを主セルに必須**。
- **注意（Stage-0）**: **H-15 が本丸**（寄り itayose・清算/信用カレンダー差で overnight 継続の符号反転/消滅が最有力
  失敗）＝両方向 2倍試行前提で K 予算・グリッド最小化。anchor 2019＝H-19（成分持続は構造的でアノマリー減衰と別と
  主張できるが減衰も併記）。成分は日次計算・保有は月次＝15bps 床側で設計。日本 common stock への transfer は
  台湾（retail 支配＝最良類似）を prior に。
- **出典**: Lou, Polk & Skouras (2019) JFE 134(1):192-213／Bogousslavsky (2021) JFE 141(1):172-194／
  Ho, Hsiao, Lo & Yang (2023) PBFJ 82 102151／Chen & Kawaguchi (2018) IJEF 10(1)（全 WebFetch 確認済）＝[IDEAS I-57]。

### ❌ buyback_execution_flow — 自社株買いの「実執行フロー」（発表でなく月次取得状況・判定済み）
- **状態**: ❌ FAIL（2026-07-05 判定。scope=`buyback_execution_flow`・**Stage-0生存→judge K=3**・docs/62 §5）。
- **結果**: 最良 bef_exec_ls **DSR0.80**（SR+1.54・raw executor バスケット）だが、**turnover十分位中立 bef_exec_sizematch で
  SR−0.07・DSR0.21**＝raw の超過は執行フロー特異でなく **size/liquidity 交絡**（買付執行企業＝大型・高流動）。**F7亜型**。
  データ源は red-team で TDnet→**EDINET 自己株券買付状況報告書**に訂正（6,510 filings/1,267社/2025-06〜2026-06・submit アンカー）。
- **収穫**: **H-22 新設**（イベント・バスケット生LSの size/liquidity 交絡＝size中立セルで帰属を確定・プラセボでは拾えない）。
  発表軸 shareholder_return❌（DSR0.24）に続き執行軸も否定＝自社株買い3軸目も棄却。**再訪条件**: D-8（EDINET 2016-2024 backfill）で
  powered 化しても size中立αが無い限り不可＝実質打ち止め。executor 抽出スクリプトは backfill 時に再利用可。
- **出典**: Clarke (2022) FRL 49, DOI 10.1016/j.frl.2022.103113（確認済・米）＝[IDEAS I-51 🧪]。

### ❌ margin_alert_event — 信用規制イベントの前後ドリフト（旧 TODO ⑤(b)・判定済み）
- **状態**: ❌ FAIL（2026-07-03 判定。scope=`margin_alert_event`・K=6・docs/53 §5）。
- **結果**: 最良 dp_start_short_h5 でも **DSR0.01**。解除ロングは**符号逆（F3）**＝解除後も下落継続
  （−566bps/10日）。指定ショートはドリフト実在だが **F8亜型**＝少数集中×特異ボラ×容量¥万単位で
  SR に変換されず。独立性は高い（ρ̄≈0〜0.1）が H-6 の通りエッジ無し。
- **収穫**: H-12（SR上限の Stage-0 概算）・H-13（規制解除≠回復）。ロング簿除外スクリーンとしての
  残存価値のみ（対象少・影響軽微）。再評価条件なし＝打ち止め。

### ⏸ governance_event_value — 東証PBR改革開示イベント条件付き value（旧 TODO #3・tail KILL・D-7待ち）
- **状態（2026-07-05・Stage-0 cheap-kill 後）**: ⏸ 保留。**解除条件＝D-7（TSE 資本コスト一覧 2024-01〜2025-04 backfill）取得**。
  red-team 2026-07-05（revise）→ K=0 Stage-0（docs/63 §5）で **in-window tail（datable transition 135件・
  2025-06/07 偏在・以降崩落）を value+size 残差×forward×placebo で cheap-kill → KILL**（全 spec で placebo 区別不能 p0.12〜0.22）。
  ❌でない＝**論文本命（2024 salience shock の低PBR×高ROE repricing・I-9）は窓外で未検定**。**H-23 新設**
  （salience ショック＝水準シフトで後日 drift 無・余波 tail は breadth 崩落＝ショック窓を取れ）。→ D-7 取得後に別 scope/prereg。
- **解禁（2026-07-05・/data-acquire D-2）**: `data/tse_capital_disclosure/disclosure_panel.parquet`
  （JPX list.xlsx・**2025-05〜2026-05 の13月次snapshot**・PIT月末アンカー・開示済/検討中 status＋update_date）。
  ⚠ **カバレッジ限界**: list.xlsx は ~13月ローリング＝**2024 salience shock（2024-01 一覧表）は未収録**
  （archive/Wayback backfill が別途要）。ローカルで検定可能なのは **Standard 市場の in-window 新規開示
  （~250件・主に 2025-06〜）＋開示済/検討中 の cross-section**（Prime は ~90%飽和＝左側打切り）。
  → **red-team で「13月・Standard tail で機構検定に足るか / 2024 backfill を先にやるか」を要判断**。
- **（旧）状態**: ⏸ 保留（2026-07-03 Stage-0 ①データ実在性で保留・trend-structure Step1 で確認）。
  静的 PBR×ROE ティルトだけなら value×quality 既試 redux（F1/F7）＝**イベント増分こそが検定対象**（I-9）。
- **補強（2026-07-04 scout）**: 出典論文は JCF forthcoming に格上げ。**Standard 市場の開示率は
  ~50%＝新規開示イベントは 2026 も継続発生中**（Prime >90% 飽和）＝取得タスクの価値上昇。
  月次リストは JPX 公表アーカイブから再構築可（surveys/2026-07-04 §B）。
- **仮説**: 「資本コストや株価を意識した経営」開示（コーポレートガバナンス報告書）を出した
  割安企業は、開示イベント後に value プレミアムの実現が加速する。
- **経済的根拠**: PBR改革レジーム（2023+）の value 復権はイベント（開示・還元策）に沿って実現する。
- **データ**: EDINET/TDnet 開示（`data/edinet/list/`・`data/tdnet/`）＋ value ファクター。
- **新規性**: value の**イベント条件付け**は scope に無い。
- **独立性**: ⚠ value そのものと共変（F1 の本丸）→ **value 残差化を設計に必須で組み込む**。
- **注意**: 「valueで説明できる部分を除いた増分」が検定対象であることを事前登録で固定。
- **出典**: D'Ercole, Wagner & Yamada, CEPR DP19971 / J. Corporate Finance 99 (2026)＝[IDEAS I-9]。
  設計指針: 効果は**低PBR×高ROE の交互作用**に載る・PBR=1 不連続でなく滑らか・開示イベントの
  タイミング増分が検定対象（scout 2026-07-03）。

### ❌ trend_structure — トレンド設計の構造比較（判定済み・認定未達）
- **状態**: ❌ FAIL（2026-07-03 判定。scope=`trend_structure`・K=6・docs/54 §5・DSR0.92＝最接近）。
- **結果**: H1 単一EMA 棄却／H2 バーベル 冗長性のみ確認／**H3 飽和tanh がベースラインを全リスク
  指標で支配**（SR・回転・maxDD・コスト耐性）だが DSR 0.92 で未達＝H-4 の認定不能領域。OOS 全セル正。
- **収穫**: H-14（改良研究は standalone DSR で裁けない）。tanh 減衰は本番 TSMOM の無料改善候補
  （人間判断・docs/54 §5）。再評価条件＝スパニング判定（I-16）導入後に差分系列を新 scope で。
- **仮説**: 本番 TSMOM（12-1 線形）は最適なトレンド設計ではない。(H1) 適切な速度の単一EMA
  （半減期~78営業日）が同等以上、(H2) 短期＋長期バーベルが等加重ラダー以上、(H3) S字減衰
  （極端シグナルの抑制）が線形以上——を同一ユニバース・同一コストで三つ巴判定。
- **経済的根拠**: 一時間尺度の平均回帰ドリフト過程なら単一EMAで十分（Valeyre 理論 R²≈0.98）／
  中間ホライズンは冗長（Etienne et al.）／トレンド→リターン写像はS字（Moskowitz et al.）。
  3論文の主張が部分的に対立＝どれが正しいかは検定で決まる（他市場証拠で方向を事前固定できる）。
- **データ**: 外部11資産＋TOPIX/N225 の日次終値（完備）。先物カーブ不要。
- **新規性**: registry の tsmom_multiasset は 12-1 のみ。設計構造の比較は未試行。
- **独立性**: momentum 代理そのもの＝**判定は本番スリーブに対する増分**（副次基準: 最良セルが
  12-1 ベースラインをネットで上回る）。
- **注意**: グリッド≤6（12-1基準・単一EMA・バーベル・S字×2程度＋等加重ラダー）。速い脚の回転
  コスト（H-3）。関数形・半減期は論文値で事前固定（H-11）。
- **出典**: [IDEAS I-4/I-5/I-6]（arXiv:2504.10914・arXiv:2510.23150・SSRN 5933974、全て確認済）。

### ❌ jump_tail_beta_xs — ジャンプテールβのクロスセクション（判定済み・符号逆）
- **状態**: ❌ FAIL（2026-07-03 判定。scope=`jump_tail_beta_xs`・K=4・docs/55 §5・**F3 符号逆**）。
- **結果**: 全セル負（主セル SR−0.67・DSR0.01）。急落月は全勝＝保険分類器としては機能するが
  日本では保険料を払う側＝米国のプレミアム符号が反転。独立性（ρ̄≈0）は残差化で達成済み。
- **収穫**: H-15（米国リスクプレミアム系は日本で符号反転しがち）。LT尺度キャッシュ再利用可。
  反転版（クラッシュ売り）は F5 隣接＝積まない。再評価条件なし＝打ち止め。
- **仮説**: N225 指数オプションの deep-OTM プットから作る左テール尺度への感応度（テールβ）が
  高い銘柄は将来劣後する（テールヘッジ需要による過大価格＝負のリスクプレミアム）。
- **経済的根拠**: クラッシュ保険を提供する資産は買われすぎる（Alexiou & Rompolis: 米で
  high-low −9.95%/年・下側テール主導）。
- **データ**: `options_225`（2016-06〜2026-06・日次2,620ファイル・IV/Strike/UnderPx）＋全銘柄日次。完結。
- **新規性**: 既試 vol_premium_n225（VRP売り）・option_regime_topix（レジームゲート）とは別物＝
  オプション情報の**クロスセクション利用**は registry 初。
- **独立性**: ⚠ 低ボラ/BAB 代理リスク（low_risk_anomaly と要直交化＝β・低ボラへの残差化を設計に含む）。
- **注意**: OTM 板の疎さ→テール尺度の推定安定性が実装リスク（EVT フィットの頑健性を §1 実現可能性
  スキャンで先に確認）。月次リバランス・コスト15bps。
- **出典**: [IDEAS I-11]（J. Empirical Finance 79 (2024)、abstract 確認済）。

### ❌ delta_liquidity — 流動性変化のアンダーリアクション（判定済み・符号逆＋momentum代理）
- **状態**: ❌ FAIL（2026-07-03 判定。scope=`delta_liquidity`・K=4・docs/56 §5・DSR0.00）。
- **結果**: 両仮説とも符号逆（F3）＋ LT 変化は momentum と ρ̄−0.46（F4 亜型＝同窓価格の同時性・
  事前登録コントロールに同窓 momentum が欠落）＋回転 1.7〜3.2回/月で gross≈0 → net 即死。
- **収穫**: H-16（Δ系シグナル＝同窓価格系の機械的代理＋構造的高回転）。
  再評価条件＝原論文全文の成分定義が当方近似と実質的に異なる場合のみ（低EV）。
- **仮説**: 流動性の**変化**（水準でない）への市場の過小反応が翌月リターンを予測する。
  方向は原論文の成分定義に忠実に固定: 固有・**長期**の流動性変化は翌月リターンに**負**・
  固有・**短期**の変化は**正**（Iwanaga & Hirose 2023 abstract 明記＝事前登録時に本文で成分定義を確認）。
- **経済的根拠**: 流動性が変わると要求プレミアムが変わるが、再価格付けは同月内で完結せず
  翌月に持ち越される（illiquidity ベースの過小反応・attention 仮説は棄却済み＝論文内）。
- **データ**: features の `amihud_illiq`・`turnover`・spread 系＝完全ローカル・月次・実装最軽量。
- **新規性**: 静的な流動性**水準**（amihud/low_vol 系）は既試だが**一階差分軸**は registry に無い。
- **独立性**: ⚠ 短期リバーサル・サイズ・流動性水準との直交化が必須（残差化を主セルに・
  残差セルの生存を副次基準に）。
- **注意**: 流動性が大きく変化する銘柄はマイクロストラクチャが不安定＝コスト実現性を診断で確認。
  日本主標本（H-15 適合）。
- **出典**: Iwanaga & Hirose, PBFJ 81 (2023) 102115 ＋ PBFJ 75 (2022)＝[IDEAS I-29]。

### ❌ activist_filing_drift — 大量保有報告イベントの filer 異質性ドリフト（判定済み）
- **状態**: ❌ FAIL（2026-07-03 判定。scope=`activist_filing_drift`・K=4・docs/57 §5・DSR0.08）。
- **結果**: filer 異質性は日本論文どおり再現（名簿filer +386bps/5日 vs 全filer 無）だが
  **ジャンプ型織り込み**＝T+1 後の残余が細く、かつ 2024+ で標本内急減衰（crowding）。
- **収穫**: H-17（開示イベントの高速織り込み）。filer名簿/重要提案フラグはスクリーン部品として有効。
  edinetCode→secCode 対応表は汎用資産。再評価条件＝日中執行が可能になった場合のみ。
- **仮説**: 大量保有報告（初回・変更）後のドリフトは filer タイプと目的で異質＝
  ファンド系 filer（ヘッジファンド/投資ファンド）・重要提案行為目的の報告後に正のドリフト、
  PE・事業会社等では弱い/無し（方向は Gillan et al. 2023 で事前固定・詳細は原文確認）。
- **経済的根拠**: 「誰が・何の目的で」買ったかの情報は開示時に一度に織り込まれず段階的に
  実現する（エンゲージメントの進行・追随買い）。
- **データ**: `data/edinet/large_holdings.parquet`（4,872件）＋`large_holdings_changes.parquet`
  （2,903件）＝filer名・目的・保有%・日付が解析済み。イベント数 300-800 見込み・保有1-3ヶ月。
- **新規性**: 既試「アクティビスト静的レジストリ」（銘柄スクリーン）とはイベント軸で別。
  registry に大量保有イベント scope は無い。
- **独立性**: ⚠ F1（アクティビストの低PBR選好＝2023+ value 共変）→ value 残差化＋サブ期間
  符号を副次基準に。
- **注意**: filer 分類（ファンド系の判定）を事前に固定（後からの分類変更は事後救済）。
  filer 母集団は広く（H-12）。日本主標本（H-15 適合）。
- **出典**: Gillan, Nguyen & Nishikawa, PBFJ 77 (2023) 101891 ＋ Yoshida, JRI (2026)＝[IDEAS I-33]。

### ❌ margin_deadline — 制度信用6ヶ月期日のコホート供給（判定済み・機構不発）
- **状態**: ❌ FAIL（2026-07-03 判定。scope=`margin_deadline`・K=4・docs/58 §5・DSR0.00）。
- **結果**: **プラセボ窓と完全同値（+8bps/週）＝期日局在なし**を一意に識別（27,063イベント・
  検定力十分）。乗り換え・早期決済で期日供給は霧散＝「期日売り」lore は週次データで不成立。
- **収穫**: H-18（統制セルの標準装備）。再評価条件＝建玉日別/日次残高データの入手時のみ（低EV）。
- **仮説**: 制度信用の買残急増（週次で観測）は約26週後の期日週に機械的な売り供給を生み、
  急増コホートの期日窓で対象銘柄はアンダーパフォームする（買残急増→期日窓ショート/回避方向）。
- **経済的根拠**: 6ヶ月以内の強制決済という**制度ルール**＝カレンダー付き需給（情報でない）。
  開示が無いためジャンプの起点が無い（H-17 適合）。信用買い手は uninformed 追随者（IRFA 2016）＝
  期日決済は情報でなく機械的供給。
- **データ**: `data/jquants/margin_weekly/`（銘柄別週次買/売残・2016-06+）＋日次 wide。完全ローカル。
- **新規性**: 期日コホート効果は**学術的に未検証**（実務 lore のみ）。既試 margin_overhang
  （静的な買残水準 XS・❌F6）とは「コホートの期日タイミング」軸で別。registry に scope 無し。
- **独立性**: ⚠ H-16＝買残の急増は同窓の価格上昇と同時（追随買い）→ 急増判定を
  ADV/浮動株比で定義し、同窓 momentum との直交化・残差化を設計に組み込む。
- **注意**: F8（高信用倍率は小型に偏る＝コスト/容量診断必須）・期日分散（乗り換え・早期決済）で
  霧散なら F5 に分類して打ち止め。週次粒度＝期日窓は±1週の幅を持たせる。
- **出典**: Hirose, Kato & Bremer (2009) PBFJ 17(1)（確認済）＋IRFA 45 (2016)＋JPX 制度規則
  ＝[IDEAS I-37]。

### ❌ unexpected_forecast — 経営者予想の「予想外部分」の過大評価（判定済み・減衰）
- **状態**: ❌ FAIL（2026-07-03 判定。scope=`unexpected_forecast`・K=4・docs/59 §5・DSR0.37）。
- **結果**: 副次基準5つ全通過・序列は論文と完全一致（DF＞FI＞NDF統制）・分位単調＝**機構は実在**。
  だが効果規模が微小（Q5−Q1 +24bps/月 gross・コスト0でも SR+0.13）＝2003-09 標本の公表後減衰（F6）。
- **収穫**: H-19（古い標本は方向のみ・大きさは年代でディスカウント）。I-44 は独立軸確認済みだが
  同ファミリーの規模から優先度低。再評価条件＝EDINET long 充実で AB 9シグナルのフル構築時のみ。
- **仮説**: 期初経営者予想のうちファンダメンタルズで説明できない unexpected 部分が**高い**銘柄は
  その後12ヶ月アンダーパフォームする（市場は less credible な裁量部分を過大評価し、期中実績で
  ゆっくり修正）。expected 部分には予測力なし。
- **経済的根拠**: 日本の実質強制の点予想には経営者裁量（forecast management）が混入する。
  市場は credible/less credible を部分的にしか区別できず、修正は年度を通じて実現（H-17 適合）。
- **データ**: fins_summary の F*（FNP/FOP/FEPS 等）＋実績＋PIT 開示日＝完全ローカル。
  予想モデル＝ラグ付きファンダメンタルズ＋前年予想誤差のクロスセクション回帰（事前登録で固定）。
- **新規性**: 既試 guidance_bias（集計タイミング・❌）は市場全体の時系列・本件は**企業横断の
  水準特性**。forecast_revision（改訂Δ）とも別。registry に scope 無し。
- **独立性**: ⚠ F7＝unexpected 楽観はディストレス/低品質と相関 → quality/ディストレス代理への
  残差化セルを主セル化＋残差セル生存を副次基準に。
- **注意**: 事前登録前に CARF F367（オープン版）で予想モデルの仕様を確認し忠実に固定すること。
  方向は論文で固定済み（unexpected 高→負）。
- **出典**: Kitagawa & Shuto (2024) JBFA 51(9-10)（RePEc abstract 確認済）＝[IDEAS I-43]。

### ❌ topix_staged_flow — TOPIX 段階的ウエイト低減の決定論的フロー（データ解禁済み・2026-10 前）
- **仮説**: 公表スケジュール済みの機械的インデックス売り（四半期最終営業日×10段階）は、
  実施日近傍に一時的価格圧力と**その後のリバーサル**を生む（明田 2022: 2022-10-28 イベントで
  −1.6%→2週間で復元）。方向・窓は明田の実測とスケジュール規則で事前固定。
- **経済的根拠**: H-17 適合の本命型＝情報でなく**カレンダー付き強制フロー**（低減幅・日付・
  対象が全て事前公知）。指値供給者への一時的プレミアム。
- **データ**: `data/jpx_indices/topix_reduction_events.parquet`（988行・**2026-07-04 調達済み**・
  検証済み＝research_ops/acquisitions/2026-07-04-jpx-topix-lists.md）＋日次 wide。
  Stage1 = 493銘柄×10実施日 ≈ 4,900 銘柄イベント。
- **新規性**: 既試 index_events_n225 は「入替ドリフト」・本件は「多段階既知フローのリバーサル」
  ＝設計が別（F7 正当化は docs 事前登録で）。
- **独立性**: 対象は流通時価総額<¥100億の小型＝サイズ軸との交絡に注意（ヘッジ設計で対処）。
- **注意**: H-18＝プラセボ窓（実施日と無関係な同コホート窓）を標準装備。H-12＝実施日毎に
  ~440-490銘柄が同時イベント＝breadth は十分。コスト床は日次イベント 30bps。
  **Stage2（2026-10 開始）＝真の前向き OOS**: リスト公表（2026-08 基準日以降）と同時に
  フォワード事前登録する二段構え。
- **出典**: [IDEAS I-32]（明田 2022 JSRI・QUICK 次世代TOPIX 規則・確認済み）。
- **判定 2026-07-04（docs/60 §5・DSR0.00 FAIL）**: gross 機構は実在・削減特異（プラセボクリーン・
  リバーサル+75bps・8/10ステップ正）だが**借株300bps＋容量¥183万で純化消失＝執行不能**（H-20）。
  実効 n=10 で認定も構造的不可。**Stage2（2026-10 前向き）は素朴小型 LS では EV 低**が確定＝
  再開するなら「大型サブセット or 執行コスト実効低減」の角度を先に設計（それが無ければ着手しない）。

### ⏸ gkx_imputation — GKX 行列の欠損値補完で再検証（旧 TODO #4・**前提タスク待ちでブロック**）
- **仮説**: phase4/4b の GKX 帰無（ML 予測が無効）は欠損値処理（NaN→0）で検定力を失った疑いがあり、
  クロスセクション補完で再検証すると結論が変わり得る。
- **データ**: `data/phase4/`・`data/phase4b/`（構築済み行列）。
- **新規性**: 手法検証（既存 scope の方法論的再訪）。K は新 scope で計上。
- **注意**: 「補完方法の選び直し」を繰り返すと p-hack 化する → 補完法は1つ事前固定。
- **⏸ 2026-07-04（red-team・operator 第3起動）**: 保存済み X.parquet は phase4/4b とも
  **NaN率0.00%＝既に補完済み dense**（ゼロ値13.4%＝補完ゼロと真ゼロが区別不能）。
  再補完の対象が保存行列に存在しない＝**生 NaN を保持した行列の再構築が前提**
  （research_ops/redteam/2026-07-04-gkx_imputation.md）。→ 下記の前提タスク完了まで着手不可。
  加えて H-14（手法改良は単独認定不可＝帰無再評価・人間サインオフ案件）を §2.6 に要明記。
- **前提タスク（データ基盤・研究仮説でない＝⬜ キュー外）**: invest_system の行列ビルダーに
  「補完前（真 NaN 保持）で emit する分岐」を追加（examples/build_phase4_raw.py 等・K=0）。
  これが解ければ NaN→0 baseline vs CS補完 の比較が実行可能。人間が着手を判断。

### ❌ sjm_per_factor_regime — per-factor regime（SJM 型・浅い状態切替）（旧 TODO #5・判定済み）
- **状態**: ❌ FAIL（2026-07-05 判定・**Stage-0 KILL・K=0**・scope=`sjm_per_factor_regime`・docs/61 §5）。
- **結果**: FF Japan 3因子・430ヶ月の best-case でも switched SR **+0.306 < 固定 +0.671**（**F5** de-risk 有害）・
  placebo(shuffle) mean+0.309/p95+0.462 の内側（**H-18** timing 情報ゼロ）・mom+0.898 に劣後（**H-1**）。
  遷移 median 5＝推定可＝**H-11 データ不足を反証**＝機構そのものが JP 因子配分で無効。
- **収穫**: **H-21 新設**（regime 切替は placebo 同値になりがち＝shuffle placebo 必須）。オンライン・ジャンプ
  モデル基盤は他の regime/timing 候補の Stage-0 に再利用可。**打ち止め**（データ増でも効かないと反証済み）。
- **出典**: Shu & Mulvey (2024) arXiv:2410.14841＝[IDEAS I-12]。red-team 2026-07-05 revise 全反映。

### ⏸ tostnet_blockflow — ToSTNeT 超大口フロー（旧 TODO ⑪）
- **解除条件**: 前向き蓄積が検定に足る件数に達すること（`data/jpx_tostnet/` 2026-06 開始・
  保持2週間分を日次追記中）。目安＝12ヶ月以上。

### ⏸ supplier_leadlag — 取引先リードラグ（旧 TODO ⑩/#6）
- **解除条件**: 顧客リンクのローカル PIT データ化（現状 EDINET-DB MCP の per-company クエリ依存＝
  全ユニバース構築が非現実的・F9）。

### ⏸ limit_reversal_jsf — ストップ高リバーサル再評価（docs/48 §5 の再評価条件）
- **解除条件**: JSF 逆日歩データ統合（`data/jsf/` 未取得）で空売り可否・実コストを実値化。
  ただし損益分岐≈25bps・容量極小は不変の見込み＝**低EV・打ち止め推奨**を継承。

---

## §2 除外リスト（既判定・再掲禁止 — 同じ経済機構＋同じデータ軸は別名でも既試）

TODO.md（凍結）から継承。詳細な判定は registry 各 scope・docs/03 §6・docs/45-48 参照:

- **ファクター系**: value/quality/profitability(VQP)・PEAD/予想改訂・残差モメンタム・低ボラ・
  BAB/IVOL(low_risk_anomaly)・短期リバーサル・マイクロキャップ逆張り・52週高値(high_52w)・
  アセットグロース(asset_growth)・DSO/収益の質(dso_quality)・信用買残オーバーハング(margin_overhang)・
  空売り残高XS(short_interest)・サイズ（生存者バイアス由来と判明）
- **イベント系**: 自社株買い＋増配(shareholder_return)・保守的予想バイアス(guidance_bias)・
  開示タイミング(disclosure_timing)・指数入替(index_events_n225)・TOB裁定(tob_arb・フォワード監視は別枠で継続)・
  アクティビスト・ストップ高リバーサル(limit_reversal, docs/48)・自社株買い実執行フロー(buyback_execution_flow, docs/62・size交絡)
- **タイミング/マクロ系**: 海外フロー→TOPIX(flow_topix_timing)・スマートマネー乖離(flow_divergence_topix)・
  オプションIVレジーム(option_regime_topix)・FX/金利×セクター(macro_sector_rotation)・
  国債ターム・キャリー(bond_term_carry)・ボラ売り/VRP(vol_premium_n225)
- **手法系**: GKX キッチンシンクML(phase4/4b)・共和分ペア(meanrev_pairs/meanrev_regime)・
  開示テキストα cheap版(disclosure_text_change, docs/45)・条件付きSDF(conditional_sdf, docs/46)・
  単一銘柄日足レジーム検知（2026-06 事前登録 negative で打ち切り・再開条件はクロスセクション化/Dollar Bars のみ）
- **柱の現行結論**: TSMOM は採用済み（§6.16 2スリーブ）。crypto(bitbank BTC/JPY 4h RF)は判定済み帰無。

## §3 Stage-0 棄却台帳（append-only — 再生成防止の dedupe 台帳）

| 日付 | 候補（slug＋1行） | 棄却理由（Stage-0 チェック番号・H-n） | 証拠 |
|---|---|---|---|
| 2026-07-05 | sjm_per_factor_regime（per-factor regime 切替＝固定超え） | F5（固定未満）＋H-18（placebo 同値＝timing 情報ゼロ）＋H-1（mom 劣後）。H-11 は反証（430月・遷移 median 5＝推定可） | docs/61 §5・K=0 |
| 2026-07-05 | governance_event_value **tail**（2025+ 新規開示の value 増分・※本命2024窓は別） | F6/H-9/H-17（salience 減衰・水準シフトで後日drift無）＋breadth 崩落（datable 135件・全spec placebo区別不能 p0.12〜0.22）→**H-23**。**⏸（scope自体は D-7=2024窓 backfill で再開）** | docs/63 §5・K=0 |
