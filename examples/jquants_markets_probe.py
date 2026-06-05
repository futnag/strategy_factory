"""信用/空売り系エンドポイントを、稼働中ベース(api.jquants.com/v2 + x-api-key)で
実地探索する。各エンドポイント×パラメータで HTTP状態・最上位キー・実フィールド名を表示。
ドキュメント(別ベース/Bearer表記)に依らず、ライブ応答を真とする。
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import jquants as jq  # noqa: E402

# 範囲(from/to)取得が効くか＝最小呼び出し回数の探索。
TRIES = [
    # margin-interest: code+date 無しで from/to だけ受け付けるか（全銘柄を範囲で）
    ("/markets/margin-interest", {"from": "20260501", "to": "20260531"}),
    ("/markets/margin-interest", {"code": "72030", "from": "20250101", "to": "20251231"}),
    # short-ratio: s33 + from/to（1セクター全期間を少回数で）
    ("/markets/short-ratio", {"s33": "0050", "from": "20260101", "to": "20260531"}),
    # short-sale-report: disc_date 範囲だけで全銘柄を取れるか
    ("/markets/short-sale-report", {"disc_date_from": "20260401", "disc_date_to": "20260531"}),
    ("/markets/short-sale-report", {"calc_date": "20260522"}),
    # margin-alert: from/to だけで取れるか
    ("/markets/margin-alert", {"from": "20260525", "to": "20260529"}),
]


def probe(base: str, path: str, params: dict, key: str) -> None:
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": key})
    tag = f"{path} {params}"
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.load(r)
        keys = list(data.keys())
        rows = data.get("data", [])
        sample = list(rows[0].keys()) if rows else "[]"
        print(f"OK  {tag}")
        print(f"      top_keys={keys}  data_len={len(rows)}")
        print(f"      fields={sample}")
        if rows:
            print(f"      row0={json.dumps(rows[0], ensure_ascii=False)[:200]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:160]
        print(f"HTTP{e.code} {tag}  {body}")
    except Exception as e:  # noqa: BLE001
        print(f"ERR  {tag}  {str(e)[:120]}")


def main() -> int:
    key = jq._api_key()
    print(f"=== ベース {jq._BASE}（x-api-key）===")
    for path, params in TRIES:
        probe(jq._BASE, path, params, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
