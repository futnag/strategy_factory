---
name: registry-ask
description: 「これもう試した？」に即答する制度記憶の照会。試行レジストリ（registry API 経由・742+試行）・docs/ 事前登録・BACKLOG 除外/棄却台帳・IDEAS・LESSONS 失敗型・HEURISTICS・ledger を横断し、判定（既試/類似既試/未踏）を scope・K・DSR・失敗型・出典付きで返す。読み取り専用＝DB直SQL禁止・どのファイルにも書かない。引数: 質問（仮説・データ軸・戦略アイデアなど自然言語）。
---

# /registry-ask — 制度記憶への照会（1起動＝1質問1回答）

新アイデアの dedupe（PLAYBOOK Step 1 で必須の突合）を、サイクル外でも会話中でも
即座にやれるようにする読み取り専用スキル。

## ガードレール（違反不可）

1. **読み取り専用**: BACKLOG・IDEAS・LESSONS・ledger・レジストリの**どれにも書かない**。
   「IDEAS に追記する価値がある」と判断したら、その旨と草稿を提示して人間（または
   /research-scout）に委ねる。
2. **`data/research_trials.db` への直接 SQL 禁止**（PLAYBOOK §A-4）。アクセスは
   `examples/loop_status.py`・`examples/registry_status.py`・
   `invest_system/validation/registry` の API 経由のみ。
3. **保守的判定**: 「同じ経済機構＋同じデータ軸」なら別名でも**既試**と判定する
   （PLAYBOOK Step 1 の基準）。迷ったら「類似既試」に倒し、相違点を明示する。
4. 記憶で答えない: 必ず当該 scope の一次記録（docs/NN §5・LESSONS・レポートHTML）を
   開いてから答える。

## 手順

### Step 1 質問の正規化
質問を {経済機構（なぜ儲かるか）× データ軸（何で測るか）× 実装形（XS/イベント/時系列/ペア）}
に分解して復唱。曖昧なら最も近い2解釈を並記して両方調べる。

### Step 2 横断検索（この順で）
1. `$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\loop_status.py` — scope 一覧・K・直近試行。
2. `research_loop/BACKLOG.md` §2 除外リスト・§3 Stage-0 棄却台帳（バックテスト無しで
   殺された候補はレジストリに居ない＝ここが唯一の記録）。
3. `research_loop/LESSONS.md`（失敗型タクソノミー＋サイクル記録）・`HEURISTICS.md`
   （Stage-0 の prior）・`IDEAS.md`（備蓄済み候補と重複しないか）。
4. ヒットした scope の一次記録: `docs/NN-<scope>-preregistration.md` §5・
   `data/reports/<scope>.html`・必要なら registry API で試行詳細。
5. どこにも無ければ `docs/24-data-catalog.md` で必要データの実在も確認（未踏でも
   データが無ければ「未踏・ただしデータ制約」と答える）。

### Step 3 回答様式（固定）
- **判定**: 既試 / 類似既試（相違点1行） / 未踏 / 未踏・データ制約
- **記録**: scope・K・最良 DSR・verdict・失敗型（LESSONS の型名で）
- **機構メモ**: なぜその結果になったか1-2行（docs/NN §5 の解釈を引用）
- **出典**: docs/NN・LESSONS 行・レポートHTML のパス
- **次アクション**: 既試→終了/再開条件があれば引用。未踏→Stage-0 で見込みがあるか
  HEURISTICS を当てた所感＋「載せるなら /research-scout → BACKLOG 経由」と案内。

## 運用メモ
- 会話の途中で「それ試したっけ？」と聞かれた時に単発で使うのが主用途。
- ヒット0でも「未踏」と断定する前に、語彙違い（英名/日本語名・factor名の別記法）で
  BACKLOG/LESSONS を grep し直すこと（除外リストは日本語・scope は英語が多い）。
