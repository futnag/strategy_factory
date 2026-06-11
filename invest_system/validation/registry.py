"""試行レジストリ（事前登録ゲート付き・研究監視）。

統合ナレッジベース §2, §5.3 / DP7, DP10 の実装。
- 事前登録（仮説＋経済的合理性）なしに結果を記録できない（a priori 理論の強制）。
- 試行は追記専用：削除・改竄 API を提供しない（"Complete / Coerced" の担保）。
- scope 単位で試行数 K と Sharpe 分散を集計し、DSR を自動算出できる。

ソロ運用での最大の敵＝自己 p-hacking を「意志」でなく「コード」で封じる中核部品。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import dsr as _dsr

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid               TEXT UNIQUE NOT NULL,
    scope              TEXT NOT NULL,
    strategy_id        TEXT,
    hypothesis         TEXT NOT NULL,
    economic_rationale TEXT NOT NULL,
    params_json        TEXT,
    status             TEXT NOT NULL,
    sharpe             REAL,
    n_obs              INTEGER,
    skew               REAL,
    kurt               REAL,
    returns_hash       TEXT,
    extra_json         TEXT,
    fingerprint        TEXT,
    preregistered_at   TEXT NOT NULL,
    completed_at       TEXT
);
"""

_MIN_TEXT = 8  # 仮説・根拠の最小文字数（空・形式的記入を拒否）


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(scope: str, strategy_id: str, params: Optional[dict]) -> str:
    """試行の指紋（scope＋戦略＋パラメータ）。同一試行の再実行を冪等にする。"""
    payload = f"{scope}|{strategy_id}|{json.dumps(params or {}, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def default_registry(path: str = "data/research_trials.db") -> "TrialRegistry":
    """永続グローバル・レジストリ（data/ 配下＝gitignore済、セッション跨ぎで累積）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return TrialRegistry(path)


class TrialRegistry:
    """SQLite を背後に持つ改竄不能な試行台帳。"""

    def __init__(self, db_path: str = "trials.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        # 既存DBに fingerprint 列が無ければ追加（前方互換）
        cols = [r["name"] for r in self._conn.execute("PRAGMA table_info(trials)")]
        if "fingerprint" not in cols:
            self._conn.execute("ALTER TABLE trials ADD COLUMN fingerprint TEXT")
        self._conn.commit()

    # --- 事前登録ゲート -------------------------------------------------
    def preregister(self, *, scope: str, hypothesis: str,
                    economic_rationale: str, strategy_id: Optional[str] = None,
                    params: Optional[dict] = None) -> str:
        """試行を事前登録し uuid を返す。仮説・経済的合理性は必須（a priori）。"""
        if len(hypothesis.strip()) < _MIN_TEXT:
            raise ValueError("hypothesis is required (state the a priori theory)")
        if len(economic_rationale.strip()) < _MIN_TEXT:
            raise ValueError("economic_rationale is required (state the a priori theory)")
        tid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO trials (uuid, scope, strategy_id, hypothesis, "
            "economic_rationale, params_json, status, preregistered_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (tid, scope, strategy_id, hypothesis.strip(),
             economic_rationale.strip(),
             json.dumps(params or {}, ensure_ascii=False),
             "preregistered", _now()),
        )
        self._conn.commit()
        return tid

    # --- 結果記録（追記専用・一度きり） --------------------------------
    def record_result(self, trial_uuid: str, *, sharpe: float, n_obs: int,
                      skew: float, kurt: float,
                      returns_hash: Optional[str] = None,
                      extra: Optional[dict] = None) -> None:
        """事前登録済み試行に結果を1回だけ記録。未登録・二重記録は拒否。"""
        row = self._conn.execute(
            "SELECT status FROM trials WHERE uuid=?", (trial_uuid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"trial not preregistered: {trial_uuid}")
        if row["status"] != "preregistered":
            raise ValueError("result already recorded (trials are immutable)")
        self._conn.execute(
            "UPDATE trials SET status='completed', sharpe=?, n_obs=?, skew=?, "
            "kurt=?, returns_hash=?, extra_json=?, completed_at=? WHERE uuid=?",
            (sharpe, n_obs, skew, kurt, returns_hash,
             json.dumps(extra or {}, ensure_ascii=False), _now(), trial_uuid),
        )
        self._conn.commit()

    # --- 冪等記録（永続グローバル運用向け） ---------------------------
    def log_trial(self, *, scope: str, strategy_id: str, params: Optional[dict],
                  sharpe: float, n_obs: int, skew: float, kurt: float,
                  hypothesis: str, rationale: str) -> str:
        """事前登録＋結果を一括記録（指紋でUPSERT＝再実行は冪等）。

        同一 (scope, strategy_id, params) の再実行は新規カウントせず結果のみ更新
        （K を水増ししない）。新パラメータは新規試行＝K を増やす。仮説・経済的
        合理性は必須（a priori 理論の強制）。返り値 uuid。

        注意：この経路の冪等性は「結果の**上書き**」で実現する（K は不変だが
        sharpe 等は最新実行で置換される）。データ期間を変えた再実行も同一指紋＝
        上書きになる。「一度きり・改竄不能」の厳密な追記専用保証が必要な試行は
        preregister + record_result（二重記録を拒否）を使うこと。
        """
        if len(hypothesis.strip()) < _MIN_TEXT:
            raise ValueError("hypothesis is required (state the a priori theory)")
        if len(rationale.strip()) < _MIN_TEXT:
            raise ValueError("economic_rationale is required (state the a priori theory)")
        fp = _fingerprint(scope, strategy_id, params)
        row = self._conn.execute(
            "SELECT uuid FROM trials WHERE scope=? AND fingerprint=?", (scope, fp)
        ).fetchone()
        if row is not None:
            self._conn.execute(
                "UPDATE trials SET sharpe=?, n_obs=?, skew=?, kurt=?, completed_at=? "
                "WHERE uuid=?", (sharpe, n_obs, skew, kurt, _now(), row["uuid"]))
            self._conn.commit()
            return row["uuid"]
        tid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO trials (uuid, scope, strategy_id, hypothesis, "
            "economic_rationale, params_json, status, sharpe, n_obs, skew, kurt, "
            "fingerprint, preregistered_at, completed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tid, scope, strategy_id, hypothesis.strip(), rationale.strip(),
             json.dumps(params or {}, ensure_ascii=False), "completed",
             sharpe, n_obs, skew, kurt, fp, _now(), _now()))
        self._conn.commit()
        return tid

    def log_scan_trials(self, *, scope: str, count: int, hypothesis: str,
                        rationale: str) -> int:
        """探索しただけ（建玉に至らない）候補を K に算入する placeholder 試行。

        ペア/バスケット探索の SBuMT（KB §11.7 / DP13）：CADF 等で事前棄却した候補も
        「検定した試行」として K に数える。sharpe=NULL ゆえ V[SR] には寄与しない。指紋で
        冪等（再実行で K を水増ししない）。返り値：今回新規に登録した件数。
        """
        if len(hypothesis.strip()) < _MIN_TEXT or len(rationale.strip()) < _MIN_TEXT:
            raise ValueError("hypothesis/economic_rationale required")
        added = 0
        for i in range(int(count)):
            fp = _fingerprint(scope, "__scan__", {"i": i})
            row = self._conn.execute(
                "SELECT uuid FROM trials WHERE scope=? AND fingerprint=?",
                (scope, fp)).fetchone()
            if row is not None:
                continue
            self._conn.execute(
                "INSERT INTO trials (uuid, scope, strategy_id, hypothesis, "
                "economic_rationale, params_json, status, sharpe, n_obs, skew, "
                "kurt, fingerprint, preregistered_at, completed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), scope, f"__scan__{i}", hypothesis.strip(),
                 rationale.strip(), "{}", "completed",
                 None, None, None, None, fp, _now(), _now()))
            added += 1
        self._conn.commit()
        return added

    # --- 集計（DSR 用） ------------------------------------------------
    def trial_count(self, scope: str) -> int:
        """scope 内の完了試行数 K（DSR の試行数）。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS k FROM trials WHERE scope=? AND status='completed'",
            (scope,),
        ).fetchone()
        return int(row["k"])

    def sharpe_variance(self, scope: str) -> float:
        """scope 内の完了試行 Sharpe の分散（DSR の sr_variance）。試行<2 は 0。"""
        rows = self._conn.execute(
            "SELECT sharpe FROM trials WHERE scope=? AND status='completed' "
            "AND sharpe IS NOT NULL", (scope,),
        ).fetchall()
        vals = [r["sharpe"] for r in rows]
        if len(vals) < 2:
            return 0.0
        return float(np.var(vals, ddof=1))

    def list_scopes(self) -> list:
        """[(scope, K, sr_variance)] 一覧（完了試行のみ）。永続レジストリの俯瞰用。"""
        rows = self._conn.execute(
            "SELECT scope, COUNT(*) AS k FROM trials WHERE status='completed' "
            "GROUP BY scope ORDER BY scope").fetchall()
        return [(r["scope"], int(r["k"]), self.sharpe_variance(r["scope"]))
                for r in rows]

    def deflated_sharpe(self, trial_uuid: str) -> float:
        """指定試行の DSR を、その scope の K と Sharpe 分散から自動算出。"""
        row = self._conn.execute(
            "SELECT scope, sharpe, n_obs, skew, kurt, status FROM trials "
            "WHERE uuid=?", (trial_uuid,),
        ).fetchone()
        if row is None:
            raise KeyError(trial_uuid)
        if row["status"] != "completed":
            raise ValueError("trial has no recorded result")
        scope = row["scope"]
        return _dsr.deflated_sharpe_ratio(
            sr=row["sharpe"], sr_variance=self.sharpe_variance(scope),
            n_trials=self.trial_count(scope), n_obs=row["n_obs"],
            skew=row["skew"], kurt=row["kurt"],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TrialRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
