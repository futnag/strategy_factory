"""指数四本値・投資部門別フローのエンドポイントを実ベースで探索（パラメータ/実フィールド確定）。"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import jquants as jq  # noqa: E402

TRIES = [
    ("/indices/bars/daily", {"date": "20260529"}),
    ("/indices/bars/daily", {"code": "0000", "from": "20260501", "to": "20260529"}),
    ("/indices/bars/daily/topix", {"date": "20260529"}),
    ("/indices/bars/daily/topix", {"from": "20260501", "to": "20260529"}),
    ("/equities/investor-types", {"date": "20260529"}),
    ("/equities/investor-types", {"section": "TSEPrime"}),
    ("/equities/investor-types", {"from": "20260501", "to": "20260529"}),
]


def probe(path: str, params: dict, key: str) -> None:
    url = f"{jq._BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": key})
    tag = f"{path} {params}"
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.load(r)
        rows = data.get("data", [])
        print(f"OK  {tag}  keys={list(data.keys())} len={len(rows)}")
        if rows:
            print(f"    fields={list(rows[0].keys())}")
            print(f"    row0={json.dumps(rows[0], ensure_ascii=False)[:200]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP{e.code} {tag}  {e.read().decode('utf-8','ignore')[:150]}")
    except Exception as e:  # noqa: BLE001
        print(f"ERR {tag}  {str(e)[:120]}")


def main() -> int:
    key = jq._api_key()
    print(f"=== ベース {jq._BASE} ===")
    for path, params in TRIES:
        probe(path, params, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
