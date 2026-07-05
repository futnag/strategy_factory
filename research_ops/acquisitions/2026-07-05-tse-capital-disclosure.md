# データ調達記録: TSE「資本コストや株価を意識した経営」開示企業一覧（tse_capital_disclosure）

**取得日時**: 2026-07-05（JST）／**タスク**: data_ideas D-2 ／**解禁**: BACKLOG governance_event_value（⏸→⬜・I-9）

## 出所・ライセンス・robots
- URL: `https://www.jpx.co.jp/equities/follow-up/jr4eth0000004vj2-att/list.xlsx`
  （ページ: `https://www.jpx.co.jp/equities/follow-up/02.html`）。
- ライセンス: JPX 公表資料（無料・公開）。**取得物は data/ 配下＝gitignore・ローカル保持のみ**（再配布しない）。
- robots.txt: `Disallow:`（空＝全許可）・follow-up パス非ブロックを実測確認。UA は連絡先明記の1リクエスト/実行。
- ※ WebFetch は UA ブロック(403)。取得スクリプトは標準 UA で正常取得（ペイウォール/ログイン回避なし）。

## 取得内容・件数
- list.xlsx（3,128,346 bytes）＝15シート。うち月末スナップショット13枚を抽出:
  **開示企業一覧（2026年5月末時点）＋【過去分】12枚（2025-05〜2026-04）**。
- panel: **29,725 行 / 2,536 社 / 13 月次断面（2025-05〜2026-05）**。
  `data/tse_capital_disclosure/disclosure_panel.parquet`＋raw `raw/list_20260705.xlsx`。

## 検証（既知事実との突合）
- 累積「開示済」社数（月×市場）:
  - **Prime ~1431→1443**（母集団~1600＝**~90% 開示で飽和**）＝TSE/報道「Prime は大半が開示」と整合。
  - **Standard 631→810**（**~50%＝新規開示が in-window で継続**）＝I-9 補強「Standard ~50%・イベント継続中」と整合。
- in-window 新規開示（初出月・左側打切り注意）: 2025-06=121・07=49・以降 8〜23/月・計 ~250（主に Standard）。
- 列検出は content ベース（プライム/スタンダード・4桁コード・「開示済」最多列＝累積 開示状況）。
  **注意**: 開示状況(累積)と「前月からの変更」列の双方に「開示済」が出るため、status は**開示済 最多列**を採用
  （変更列の取り違えを sanity check `>=100` で防止）。

## PIT 注意
- 各行のアンカー = **当該月末スナップショット**（原本 sheet 由来）。現在一覧からの遡及構成なし＝**PIT 安全**。
- 初出月（min month で開示済）= 開示イベントの近似（**左側打切り**＝2025-05 以前開示は censored）。
- `update_date`（開示内容アップデート日）は per-company の finer 日付（~65%非欠損）＝補助アンカー。

## カバレッジ限界・フォローアップ
- list.xlsx は **~13ヶ月ローリング**＝ローカルは 2025-05+ のみ。**2024 salience shock（2024-01 一覧表・I-9 の本丸）は未収録**。
  → **D-7**（archive/Wayback backfill 2024-01〜2025-04）を data_ideas 待ち行列に追加。
- スクリプトは冪等・(month,code) マージ＝**前向きに毎月深くなる**（JPX が古月を落としても手元は残る）。
- governance_event_value は red-team で「13月・Standard tail で足るか / 2024 backfill 先行か」を要判断。

## 再現コマンド
```
$env:PYTHONUTF8="1"; .venv/Scripts/python.exe examples/update_tse_capital_disclosure.py
```
（毎回 list.xlsx を再取得し全 snapshot を抽出→ローカル panel にマージ。夜間ワークフロー化は人間判断。）
