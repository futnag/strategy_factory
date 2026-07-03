# BACKLOG — 仮説キューと dedupe 台帳

**運用ルール**: `/research-cycle` は §1 の最優先 ⬜ を1つ取って1サイクルを回す（1サイクル1仮説）。
新候補は**仮説／経済的根拠／データ／新規性／独立性／注意**の構造で追記。着手時 ⬜→🔬、
判定で ✅/❌、保留は ⏸（解除条件を明記）。dedupe は §2（既判定）＋§3（Stage-0 棄却）＋
registry の scope 一覧（`loop_status.py`）の3つに対して行う。

**状態凡例**: ⬜ 未着手 / 🔬 検証中 / ✅ PASS / ❌ FAIL / ⏸ 保留

---

## §1 キュー（優先順）

### 🔬 margin_alert_event — 信用取引の日々公表・規制入りイベントスタディ（旧 TODO ⑤(b)）
- **状態**: 🔬 検証中（2026-07-03 事前登録＝docs/53・サイクル 2026-07-03-margin-alert-event）
- **仮説**: 日々公表銘柄指定・増担保規制の指定/解除の前後に予測可能なドリフト（スクイーズ→反転）。
- **経済的根拠**: 規制イベントは信用需給の強制的な変化を生む（新規買い禁止→買い圧力消滅）。
- **データ**: `data/jquants/margin_alert/`（日次・規制銘柄）＋ wide 株価。ローカル完備。
- **新規性**: ⑤(a) 買残オーバーハング（XS・❌F6）とは別物の**離散イベント軸**。registry に scope 無し。
- **独立性**: イベントアンカー＝momentum/value と構造的に低相関の見込み（H-10）。
- **注意**: 対象銘柄は小型・高ボラに偏る → H-3 コスト床（日次 30bps・貸株 300bps）を厳守。
  ロング脚（解除後）中心の設計も検討（H-7）。E2E 初回サイクルの推奨候補。

### ⬜ governance_event_value — 東証PBR改革開示イベント条件付き value（旧 TODO #3）
- **仮説**: 「資本コストや株価を意識した経営」開示（コーポレートガバナンス報告書）を出した
  割安企業は、開示イベント後に value プレミアムの実現が加速する。
- **経済的根拠**: PBR改革レジーム（2023+）の value 復権はイベント（開示・還元策）に沿って実現する。
- **データ**: EDINET/TDnet 開示（`data/edinet/list/`・`data/tdnet/`）＋ value ファクター。
- **新規性**: value の**イベント条件付け**は scope に無い。
- **独立性**: ⚠ value そのものと共変（F1 の本丸）→ **value 残差化を設計に必須で組み込む**。
- **注意**: 「valueで説明できる部分を除いた増分」が検定対象であることを事前登録で固定。

### ⬜ gkx_imputation — GKX 行列の欠損値補完で再検証（旧 TODO #4）
- **仮説**: phase4/4b の GKX 帰無（ML 予測が無効）は欠損値処理（NaN→0）で検定力を失った疑いがあり、
  クロスセクション補完で再検証すると結論が変わり得る。
- **データ**: `data/phase4/`・`data/phase4b/`（構築済み行列）。
- **新規性**: 手法検証（既存 scope の方法論的再訪）。K は新 scope で計上。
- **注意**: 「補完方法の選び直し」を繰り返すと p-hack 化する → 補完法は1つ事前固定。

### ⬜ sjm_per_factor_regime — per-factor regime（SJM 型・浅い状態切替）（旧 TODO #5）
- **仮説**: ファクター毎に regime を推定し配分を切り替えると、固定 50/50 合成を上回る。
- **注意**: H-11（小データで複雑モデルは縮退）・H-5（de-risk 形状）に正面から抵触し得る低EV枠。
  買い持ち超過＋固定合成超過を副次基準に。

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
  アクティビスト・ストップ高リバーサル(limit_reversal, docs/48)
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
| （まだ記録なし） | | | |
