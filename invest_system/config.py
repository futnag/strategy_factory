"""設定・認証情報の読み込み（.env / 環境変数）。

秘密情報（APIキー・トークン・パスワード）はコード・git・ログに書かない。
.env（gitignore済）に置き、ここで環境変数へ読み込む。依存を増やさない軽量実装。
"""
from __future__ import annotations

import os
from pathlib import Path

# リポジトリ root（invest_system/ の親）。CWD が root 以外でも .env を解決するため。
_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str = ".env") -> None:
    """.env を環境変数へ読み込む（既存の環境変数は上書きしない）。

    path（既定＝CWD の .env）が無ければリポジトリ root 直下の .env へフォール
    バックする（スクリプトを root 以外から実行しても秘密情報が読める。
    jquants の _MIN_INTERVAL 等は import 時に読むため、ここが CWD 依存だと
    実行場所によって設定が黙って既定値に落ちる）。
    """
    p = Path(path)
    if not p.exists():
        p = _ROOT / ".env"
        if not p.exists():
            return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def get_env(name: str, default: str | None = None) -> str | None:
    """環境変数を取得（必要なら .env を読み込む）。"""
    load_env()
    return os.environ.get(name, default)
