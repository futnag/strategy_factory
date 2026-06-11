"""判定結果の HTML レポート（依存追加なし・インラインSVGでエクイティカーブ描画）。

matplotlib 等に依存せず、純粋な文字列でスタンドアロンHTMLを生成する。各戦略の
ネットリターンから累積（エクイティ）曲線を SVG スパークラインで描き、判定テーブル
（SR/PSR/DSR/minTRL/容量/最大DD）と PASS/FAIL バナーを出す。
"""
from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd

from .judge import GridVerdict, _fmt_cap


def _equity(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    return (1.0 + r).cumprod()


def _svg_line(cum: pd.Series, width: int = 260, height: int = 54,
              color: str = "#1a7f5a", max_pts: int = 300) -> str:
    if cum is None or len(cum) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    y = cum.to_numpy(dtype=float)
    if len(y) > max_pts:
        y = y[np.linspace(0, len(y) - 1, max_pts).astype(int)]
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    rng = (hi - lo) or 1.0
    pad, n = 4, len(y)
    xs = [pad + i * (width - 2 * pad) / (n - 1) for i in range(n)]
    ys = [height - pad - (v - lo) / rng * (height - 2 * pad) for v in y]
    pts = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in zip(xs, ys))
    base = height - pad - (1.0 - lo) / rng * (height - 2 * pad)   # 損益分岐(=1)
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<line x1="{pad}" y1="{base:.1f}" x2="{width - pad}" y2="{base:.1f}" '
            f'stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.5"/></svg>')


_CSS = """
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;margin:24px;color:#222}
h1{font-size:20px;margin:0 0 4px} .meta{color:#666;font-size:13px;margin-bottom:14px}
.banner{padding:10px 14px;border-radius:8px;font-weight:600;margin:12px 0}
.fail{background:#fdecec;color:#a11} .pass{background:#e8f7ee;color:#137a3a}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 9px;border-bottom:1px solid #eee;text-align:right}
th{color:#666;font-weight:600;border-bottom:2px solid #ddd}
td.name{text-align:left;font-family:ui-monospace,monospace}
td.dsr{font-weight:700} .dsr.pass{color:#137a3a} .dsr.mid{color:#b8860b} .dsr.bad{color:#a11}
.note{color:#777;font-size:12px;margin-top:10px}
"""


def to_html(verdict: GridVerdict) -> str:
    """GridVerdict → スタンドアロンHTML文字列。"""
    def esc(s):
        return _html.escape(str(s))

    rows = []
    for v in verdict.results:
        cum = _equity(verdict.series.get(v.name, pd.Series(dtype="float64")))
        cls = ("pass" if v.dsr >= verdict.dsr_threshold
               else "mid" if v.dsr >= 0.5 else "bad")
        color = {"pass": "#137a3a", "mid": "#b8860b", "bad": "#a11"}[cls]
        mtrl = "∞" if np.isinf(v.min_trl) else f"{v.min_trl:.0f}"
        rob = "—" if np.isnan(getattr(v, "robustness", float("nan"))) \
            else f"{v.robustness:.2f}"
        rows.append(
            f"<tr><td class='name'>{esc(v.name)}</td><td>{_svg_line(cum, color=color)}</td>"
            f"<td>{v.sr_ann:+.2f}</td><td>{v.psr:.2f}</td>"
            f"<td class='dsr {cls}'>{v.dsr:.2f}</td><td>{rob}</td><td>{mtrl}</td>"
            f"<td>{esc(_fmt_cap(v.capacity_jpy))}</td><td>{v.max_dd:.1%}</td></tr>")
    if verdict.passed and verdict.best is not None:
        banner = (f"<div class='banner pass'>✅ PASS — {esc(verdict.best.name)} "
                  f"(DSR={verdict.best.dsr:.3f} ≥ {verdict.dsr_threshold})</div>")
    elif verdict.best is not None:
        banner = (f"<div class='banner fail'>❌ FAIL — 最良 {esc(verdict.best.name)} "
                  f"でも DSR={verdict.best.dsr:.2f} &lt; {verdict.dsr_threshold}</div>")
    else:
        banner = "<div class='banner fail'>❌ FAIL — 有効な戦略なし</div>"
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>判定: {esc(verdict.scope)}</title><style>{_CSS}</style></head><body>
<h1>判定レポート: {esc(verdict.scope)}</h1>
<div class="meta">仮説: {esc(verdict.hypothesis)}<br>
試行数 K（累計）= <b>{verdict.k}</b>, 試行間SR分散 V[SR]={verdict.sr_var:.4f},
判定基準 DSR ≥ {verdict.dsr_threshold}</div>
{banner}
<table><thead><tr><th class="name">strategy</th><th>エクイティ曲線</th>
<th>SR(ann)</th><th>PSR(&gt;0)</th><th>DSR</th><th>頑健</th><th>minTRL</th><th>容量</th>
<th>maxDD</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="note">緑=損益分岐(=1)。DSRは scope の累計試行 K={verdict.k} でデフレート
（試行を増やすほど基準が上がる＝p-hack不能）。</div>
</body></html>"""


def write_html(verdict: GridVerdict, path: str) -> str:
    """HTMLをファイルへ書き出しパスを返す。"""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_html(verdict), encoding="utf-8")
    return str(p)
