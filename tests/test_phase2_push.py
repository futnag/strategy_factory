"""Supabase 同期スクリプトの純関数部（エンコード・行マッピング）の検証。DB 不要。"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "phase2_push_supabase",
    Path(__file__).resolve().parent.parent / "examples" / "phase2_push_supabase.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["phase2_push_supabase"] = _mod
_spec.loader.exec_module(_mod)


def test_encode_decode_roundtrip_text_and_binary(tmp_path):
    t = tmp_path / "a.json"
    t.write_text('{"x": "日本語"}', encoding="utf-8")
    content, enc = _mod.encode_file(t)
    assert enc == "utf-8" and _mod.decode_file(content, enc) == t.read_bytes()
    b = tmp_path / "b.parquet"
    raw = bytes(range(256))
    b.write_bytes(raw)
    content2, enc2 = _mod.encode_file(b)
    assert enc2 == "base64" and _mod.decode_file(content2, enc2) == raw


def test_order_rows_switch_and_tsmom():
    eq = pd.DataFrame({"code": ["18020", "N225M"], "weight": [0.01, -0.8],
                       "price": [3245.0, 63160.0], "shares": [2, -1],
                       "yen": [6490.0, -631600.0], "instruction": ["買い", "売建"]})
    rows = _mod.order_rows("2026-05", "switch", eq)
    assert rows[0] == ("2026-05", "switch", "18020", 0.01, 3245.0, 2.0, 6490.0, "買い")
    ts = pd.DataFrame({"asset": ["sp500", "nk225_fut"], "weight": [0.06, 0.035],
                       "target_yen": [17720.0, 10610.0], "lots": [float("nan"), 0.0],
                       "instruction": ["記帳", "225マイクロ"]})
    rows2 = _mod.order_rows("2026-05", "tsmom", ts)
    assert rows2[0][4] is None and rows2[0][5] is None      # price/lots なし
    assert rows2[1][5] == 0.0
    assert rows2[0][6] == 17720.0


def test_num_nan_to_none():
    assert _mod._num(float("nan")) is None
    assert _mod._num(1.5) == 1.5
