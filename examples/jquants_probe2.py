"""決算発表予定 / 売買内訳のエンドポイント・アクセス可否とフィールドを実地確認。"""
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
    ("/derivatives/bars/daily/options/225", {"date": "20260529"}),
    ("/derivatives/bars/daily/options/225", {"date": "20260529", "from": "20260501",
                                             "to": "20260529"}),
]


def probe(path, params, key):
    url = f"{jq._BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": key})
    tag = f"{path} {params}"
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.load(r)
        rows = data.get("data", [])
        print(f"OK   {tag}  len={len(rows)}")
        if rows:
            print(f"     fields={list(rows[0].keys())}")
            print(f"     row0={json.dumps(rows[0], ensure_ascii=False)[:220]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP{e.code} {tag}  {e.read().decode('utf-8','ignore')[:140]}")
    except Exception as e:  # noqa: BLE001
        print(f"ERR  {tag}  {str(e)[:120]}")


def main() -> int:
    key = jq._api_key()
    print(f"=== ベース {jq._BASE} ===")
    for path, params in TRIES:
        probe(path, params, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
