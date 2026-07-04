# P-2 調査報告: Phase 2 成果物間の符号乖離 — 根本原因特定（2026-07-04）

## 結論（1段落）

**月次会計（months.csv / status.json / report）側のバグ**。`phase2_reconcile.py` L144 の
`ext_asof = ext_cl.asof(val_date)` が原因で、pandas の `DataFrame.asof` は
「**全列が非NaNである最後の行**」を返すため、外部価格パネルに1列でも古い系列があると
**ヘッジ（nk225_fut）と TSMOM 全資産の評価が古い共通日へ巻き戻る**。日次カーブ
（equity_daily.csv）は `daily_pnl_curve` が列ごとに ffill するため**正しい**。
docstring の「確定値は月次会計が正」は現状**逆**＝誤った側が正と宣言されている。

## 証拠（全て再現可能）

1. **同一実行内の乖離が再現**: 2026-07-04 14:03 の stateless 再実行でも
   months: ret_eq **+2.30%** / combo_net **+1.11%** vs カーブ: eq **−5.41%** / combo **−4.09%**。
2. **レッグ別突合**（月次パスと日次パスをコード忠実に並走）:
   株 +3,215 vs +3,215（一致）・**ヘッジ +10,600 vs −35,650（乖離の全額）**・TS −2,304 vs −2,846。
3. **asof の巻き戻り実測**: `ext_cl` の全列非NaN最終日＝**2026-06-08**
   → `asof(2026-07-03)` は nk225_fut を **65,215（6/8値）** で評価（真値 69,840・7/3）。
   原因列: `data/investers/` には Phase 2 の11資産以外の系列（dow, us500, nickel,
   aluminum, palladium, topix_fut 等）が同居しており、`update_external.py` は
   **PHASE2_KEYS の11本しか更新しない**ため、これらが 6/8-6/9 で停止
   →以後、月次会計は**恒常的に**古いマークで計算されている。
4. 2026-06-17 の初出乖離も同機構（当日は nk225_fut 障害の暫定対応中＝settings 履歴）。

## 影響（重要）

- **キルスイッチが約1ヶ月間、古いマークで判定されていた**: status.json は
  cum_net +1.11% / DD 0.0 / OK を表示しているが、真値は combo **≈ −4.1%**。
  現時点では真値でも alert 閾値（−8%）未達＝**即時の運用アクションは不要**だが、
  下落局面ではキルスイッチが鳴らないリスクだった。
- 夜間 GitHub Actions（ops リポ）も同じスクリプトを実行＝**同じバグでダッシュボードに
  誤値を配信中**の可能性が高い。
- /strategy-monitor は equity_daily（正しい側）を読んでいたため、6月の amber 判定は妥当。
- /ops-review の reconcile FAIL 検知は正しく機能した（本調査の起点）。

## 修正案（1行クラス・phase2_reconcile.py L144）

```python
# 現状（バグ）: 全列非NaN行へ巻き戻る
ext_asof = ext_cl.asof(val_date)
# 修正: 列ごとの最終既知値（daily_pnl_curve と同じ意味論）
ext_asof = ext_cl.ffill().asof(val_date)
```
追加で推奨: 必要列（nk225_fut＋TS資産）の最終有効日が val_date−5営業日より古い場合は
既存の DATA-ERROR 経路（exit 2→Issue）に流す鮮度ガード。
**適用先はローカルと ops リポの両方**。修正後に `phase2_reconcile.py` を再実行し、
months と equity_daily の一致（±cost_frac）を確認すること。

## 副次的発見（別対応可）

- `data/investers/` の非 Phase 2 系列（12本）が 6/8 以降更新停止＝更新対象の明確化
  （update_external の対象外なら panel から分離 or ドキュメント化）を推奨。
- 実施した操作: `phase2_reconcile.py` の再実行（ステートレス設計の公式月次手順・
  成果物は再生成済み。**status.json は修正が入るまで誤値 +1.11% のまま**である点に注意）。
