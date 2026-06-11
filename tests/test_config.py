"""config.load_env の .env 解決（CWD → リポジトリ root フォールバック）を検証。"""
import os

import invest_system.config as config


def test_load_env_reads_cwd_and_does_not_override(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        'FOO_TEST_KEY="abc"\n# comment\nBAR_TEST_KEY=keep-me-out\n',
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    monkeypatch.setenv("BAR_TEST_KEY", "existing")
    try:
        config.load_env()
        assert os.environ["FOO_TEST_KEY"] == "abc"      # クォート除去
        assert os.environ["BAR_TEST_KEY"] == "existing"  # 既存は上書きしない
    finally:
        os.environ.pop("FOO_TEST_KEY", None)


def test_load_env_falls_back_to_repo_root(tmp_path, monkeypatch):
    # CWD に .env が無くても、リポジトリ root の .env を読む（実行場所非依存）。
    root = tmp_path / "root"
    root.mkdir()
    (root / ".env").write_text("BAZ_TEST_KEY=42\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(config, "_ROOT", root)
    monkeypatch.delenv("BAZ_TEST_KEY", raising=False)
    try:
        config.load_env()
        assert os.environ["BAZ_TEST_KEY"] == "42"
    finally:
        os.environ.pop("BAZ_TEST_KEY", None)


def test_load_env_noop_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_ROOT", tmp_path)       # .env なし
    config.load_env()                                    # 例外なく無視される
