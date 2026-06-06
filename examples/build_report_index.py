"""研究ダッシュボード：永続レジストリの全 scope と判定HTMLへのリンクを index.html に集約。"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.validation.registry import default_registry  # noqa: E402

_CSS = ("body{font-family:system-ui,sans-serif;margin:24px;color:#222}"
        "table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid #eee;"
        "text-align:right}td.s,th.s{text-align:left}a{color:#1558d6}")


def main() -> int:
    rep = Path("data/reports")
    rep.mkdir(parents=True, exist_ok=True)
    with default_registry() as reg:
        scopes = reg.list_scopes()
    rows = []
    for scope, k, srv in scopes:
        f = rep / f"{scope}.html"
        name = (f'<a href="{html.escape(scope)}.html">{html.escape(scope)}</a>'
                if f.exists() else html.escape(scope))
        rows.append(f"<tr><td class='s'>{name}</td><td>{k}</td><td>{srv:.4f}</td></tr>")
    total = sum(k for _, k, _ in scopes)
    doc = (f"<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
           f"<title>検証ファクトリ ダッシュボード</title><style>{_CSS}</style></head><body>"
           f"<h1>検証ファクトリ — 研究ダッシュボード</h1>"
           f"<p>永続グローバル・レジストリ：{len(scopes)} scope / 総試行 {total}。"
           f"DSRは各 scope の累計試行Kでデフレート（試行を増やすほど基準が上がる）。</p>"
           f"<table><thead><tr><th class='s'>scope（判定レポート）</th>"
           f"<th>K(累計試行)</th><th>V[SR]</th></tr></thead><tbody>"
           f"{''.join(rows)}</tbody></table>"
           f"<p>各リンク＝判定レポート(HTML)。詳細は docs/03-research-findings.md。</p>"
           f"</body></html>")
    out = rep / "index.html"
    out.write_text(doc, encoding="utf-8")
    print(f"書き出し: {out}  （{len(scopes)} scope）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
