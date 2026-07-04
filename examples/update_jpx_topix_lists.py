"""JPX TOPIX 段階的ウエイト低減リストの取得・正規化（/data-acquire・I-32 解禁）。

一次ソース（公開・URL は凍結記録）:
  - 2022-10-07 公表「段階的ウエイト低減銘柄一覧」PDF（初期コホート 493 銘柄）
    https://www.jpx.co.jp/news/6030/nlsgeu000006mh6p-att/topix_j.pdf
  - 2023-10-06 公表 再評価 PDF（**2部構成・行番号の再スタートで分割**）:
    (1) 除外となる銘柄＝低減リストから外れ復帰（43銘柄）
    (2) 継続する銘柄＝逓減継続→2025-01 最終営業日に TOPIX 除外（439銘柄）
    https://www.jpx.co.jp/markets/indices/topix/tvdivq00000030ne-att/deta02_j.pdf
  - 初期493 −（43+439）＝11銘柄は再評価前に上場廃止等で消滅（derived・reeval_attrition）
  - 現行構成銘柄ウェイト CSV（検証・第2段階の前向き監視用）
    https://www.jpx.co.jp/automation/markets/indices/topix/files/topixweight_j.csv

PIT 規律: 各行動の公知日は公表日（pub_date）。低減の実施日は「四半期最終営業日×10段階
（2022-10 末〜2025-01 末）」の規則ベース＝スケジュールは公表時点で全て既知（docs/24 参照）。
出力: data/jpx_indices/topix_reduction_events.parquet [pub_date, action, code, code4, source]
      data/jpx_indices/topixweight_latest.parquet
実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\update_jpx_topix_lists.py [--refresh]
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

RAW = ROOT / "data" / "jpx_indices" / "raw"
OUT = ROOT / "data" / "jpx_indices"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SOURCES = {
    "20221007_reduction_initial.pdf":
        "https://www.jpx.co.jp/news/6030/nlsgeu000006mh6p-att/topix_j.pdf",
    "20231006_reeval_excluded.pdf":
        "https://www.jpx.co.jp/markets/indices/topix/tvdivq00000030ne-att/deta02_j.pdf",
    "topixweight_current.csv":
        "https://www.jpx.co.jp/automation/markets/indices/topix/files/topixweight_j.csv",
}
EXPECT_INITIAL = 493   # JPX 公表値（行番号 1..493 連番＝一次ソース内で自己検証）
EXPECT_ESCAPED = 43    # 再評価 (1) 除外となる銘柄＝復帰
EXPECT_CONTINUED = 439  # 再評価 (2) 継続する銘柄＝2025-01 TOPIX 除外


def fetch(refresh: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dst = RAW / name
        if dst.exists() and not refresh and dst.stat().st_size > 1000:
            print(f"  skip（既存）: {name}")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            dst.write_bytes(r.read())
        print(f"  取得: {name} ← {url}（{dst.stat().st_size:,}B）")


def parse_sections(pdf: Path) -> list[list[str]]:
    """行番号の再スタート（n_i ≤ n_{i−1}）でセクション分割し、各セクションのコード列を返す。
    各セクション内で行番号 1..N の完全連番を要求（＝一次ソース内の自己検証）。"""
    from pypdf import PdfReader
    rows: list[tuple[int, str]] = []
    for p in PdfReader(pdf).pages:
        for m in re.finditer(r"(?m)^\s*(\d{1,3})\s+(\d{4}[A-Z0-9]?)\b",
                             p.extract_text() or ""):
            rows.append((int(m.group(1)), m.group(2)))
    sections: list[list[tuple[int, str]]] = [[]]
    for n, c in rows:
        if sections[-1] and n <= sections[-1][-1][0]:
            sections.append([])
        sections[-1].append((n, c))
    out = []
    for i, sec in enumerate(sections):
        nums = [n for n, _ in sec]
        if nums != list(range(1, len(sec) + 1)):
            raise ValueError(f"{pdf.name} セクション{i + 1}: 行番号が非連番（{nums[:3]}..{nums[-3:]}）")
        codes = [c for _, c in sec]
        if len(set(codes)) != len(codes):
            raise ValueError(f"{pdf.name} セクション{i + 1}: 重複コード")
        out.append(codes)
    return out


def to5(code4: str) -> str:
    return code4 + "0" if len(code4) == 4 else code4


def main(argv: list[str]) -> int:
    print("=== JPX TOPIX 段階的ウエイト低減リスト 取得 ===")
    fetch(refresh="--refresh" in argv)

    (initial,) = parse_sections(RAW / "20221007_reduction_initial.pdf")
    escaped, continued = parse_sections(RAW / "20231006_reeval_excluded.pdf")
    if len(initial) != EXPECT_INITIAL:
        raise ValueError(f"初期コホート {len(initial)} ≠ 公表値 {EXPECT_INITIAL}")
    if len(escaped) != EXPECT_ESCAPED or len(continued) != EXPECT_CONTINUED:
        raise ValueError(f"再評価 (1)={len(escaped)}/(2)={len(continued)} ≠ 期待 "
                         f"{EXPECT_ESCAPED}/{EXPECT_CONTINUED}")
    if set(escaped) & set(continued):
        raise ValueError("再評価 (1) と (2) に重複")
    outside = (set(escaped) | set(continued)) - set(initial)
    if outside:
        print(f"  ⚠ 再評価リストに初期コホート外のコード {sorted(outside)}"
              "（2022-10→2023-10 の証券コード変更/持株会社化の承継とみられる・"
              "一次ソースのまま記録＝要データノート）")
    attrition = sorted(set(initial) - set(escaped) - set(continued))
    print(f"  初期コホート={len(initial)}  復帰(1)={len(escaped)}  "
          f"低減継続→除外(2)={len(continued)}  自然減(derived)={len(attrition)}")

    rows = ([{"pub_date": "2022-10-07", "action": "reduction_start", "code4": c,
              "source": "20221007_reduction_initial.pdf"} for c in initial]
            + [{"pub_date": "2023-10-06", "action": "reeval_escaped", "code4": c,
                "source": "20231006_reeval_excluded.pdf#sec1"} for c in escaped]
            + [{"pub_date": "2023-10-06", "action": "continue_to_removal", "code4": c,
                "source": "20231006_reeval_excluded.pdf#sec2"} for c in continued]
            + [{"pub_date": "2023-10-06", "action": "reeval_attrition", "code4": c,
                "source": "derived(set_difference)"} for c in attrition])
    ev = pd.DataFrame(rows)
    ev["pub_date"] = pd.to_datetime(ev["pub_date"])
    ev["code"] = ev["code4"].map(to5)
    ev.to_parquet(OUT / "topix_reduction_events.parquet")
    print(f"  保存: topix_reduction_events.parquet（{len(ev)}行）")

    w = pd.read_csv(RAW / "topixweight_current.csv", encoding="cp932")
    w.columns = ["date", "name", "code4", "sector", "weight", "new_index_class"]
    w = w[pd.to_numeric(w["date"], errors="coerce").notna()]  # 脚注行（（注）等）を除外
    w["date"] = pd.to_datetime(w["date"].astype(str), format="%Y%m%d")
    w["code"] = w["code4"].astype(str).map(to5)
    w["weight"] = w["weight"].astype(str).str.rstrip("%").astype(float) / 100.0
    w.to_parquet(OUT / "topixweight_latest.parquet")
    print(f"  保存: topixweight_latest.parquet（{len(w)}行・{w['date'].iloc[0]:%Y-%m-%d} 時点・"
          f"ウェイト計={w['weight'].sum():.3f}）")

    # ローカル価格パネルとの突合（検証）
    try:
        from invest_system.data.store import load_wide
        cols = set(map(str, load_wide("close").columns))
        hit = sum(1 for c in ev["code"].unique() if c in cols)
        print(f"  検証: コホート {ev['code'].nunique()}銘柄中 {hit}銘柄が wide パネルに存在"
              f"（{hit / ev['code'].nunique():.0%}）")
    except Exception as e:  # noqa: BLE001
        print(f"  （パネル突合スキップ: {e}）")
    print("PIT 注意: 実施スケジュールは規則ベース（四半期最終営業日×10段階・2022-10〜2025-01）。"
          "\n第2段階（2026-10 開始）の対象リストは 2026-08 基準日以降の公表＝公表され次第この"
          "スクリプトに URL を凍結追加する。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
