"""設定・認証情報の読み込み（.env / 環境変数）。

秘密情報（APIキー・トークン・パスワード）はコード・git・ログに書かない。
.env（gitignore済）に置き、ここで環境変数へ読み込む。依存を増やさない軽量実装。
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """.env を環境変数へ読み込む（既存の環境変数は上書きしない）。"""
    p = Path(path)
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
