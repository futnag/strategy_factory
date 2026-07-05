"""update_tse_capital_disclosure — TSE「資本コストや株価を意識した経営」開示企業一覧の取得.

出所: JPX 公表 https://www.jpx.co.jp/equities/follow-up/02.html の list.xlsx（公開・無料・ToS 準拠）。
list.xlsx は「開示企業一覧（YYYY年M月末時点）」＋【過去分】の**月末スナップショット**を各シートに保持
（現状 ~13ヶ月ローリング）。各実行で全スナップショットを抽出し、ローカル panel に**冪等マージ**して
月次履歴を蓄積する（JPX が古い月を落としても手元には残る＝前向きに深くなる）。

PIT: 各行のアンカーは**スナップショット月末**（sheet 名由来）＝「その月末時点で開示済か」。
現在の一覧から過去を推定していない（各月末の原本スナップショットを使用）＝PIT 安全。
初出月（min month で 開示済）＝開示イベントの近似（最古スナップショット 2025-05 以前開示は左側打切り）。

解禁: research_loop/BACKLOG ⏸ governance_event_value（IDEAS I-9）。
実行: $env:PYTHONUTF8="1"; .venv/Scripts/python.exe examples/update_tse_capital_disclosure.py
"""
import os
import re
import io
import time
import urllib.request as u
from datetime import datetime, timezone, timedelta
import pandas as pd

URL = "https://www.jpx.co.jp/equities/follow-up/jr4eth0000004vj2-att/list.xlsx"
UA = {"User-Agent": "Mozilla/5.0 (research data acquisition; contact futoshinagata@gmail.com)"}
OUT = "data/tse_capital_disclosure"
RAW = os.path.join(OUT, "raw")
PANEL = os.path.join(OUT, "disclosure_panel.parquet")
JST = timezone(timedelta(hours=9))
SHEET_RE = re.compile(r"開示企業一覧（(\d{4})年(\d{1,2})月末時点）")


def download(retries=3):
    os.makedirs(RAW, exist_ok=True)
    last = None
    for i in range(retries):
        try:
            with u.urlopen(u.Request(URL, headers=UA), timeout=90) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed: {last}")


def norm_local(sec_code) -> str | None:
    """証券コード → J-Quants 5桁 LocalCode（4桁は末尾0付与・英数コードはそのまま5桁化）."""
    s = str(sec_code).strip()
    if s in ("", "nan", "None"):
        return None
    s = s.split(".")[0]
    if re.fullmatch(r"\d{4}", s):
        return s + "0"
    if re.fullmatch(r"[0-9A-Za-z]{4}", s):
        return s + "0"
    if re.fullmatch(r"[0-9A-Za-z]{5}", s):
        return s
    return None


def detect_and_parse(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """content ベースで列を検出（merged header・列ずれに頑健）.

    列意味論（実測・inspect）: 開示状況(col6・累積: 開示済~2200/検討中~30-70) と
    前月からの変更(col7・8-128) の**両方に「開示済」が出る**ため、status_col は
    「開示済」出現数の**最大**列（＝累積 開示状況）を採る（変更列の取り違え防止）。
    """
    n = df.shape[1]
    market_col = code_col = None
    disc_counts = []
    for j in range(n):
        col = df.iloc[:, j].astype(str)
        if col.isin(["プライム", "スタンダード"]).sum() >= 5:
            market_col = j
        if col.str.fullmatch(r"\d{4}").sum() >= 20:      # 4桁証券コードが多数
            code_col = j
        disc_counts.append(int(col.str.contains("開示済", na=False).sum()))
    if code_col is None or market_col is None:
        raise ValueError(f"{month:%Y-%m}: 列検出失敗 (code={code_col} market={market_col})")
    status_col = int(pd.Series(disc_counts).idxmax())     # 累積 開示状況 = 開示済 が最多の列
    if disc_counts[status_col] < 100:                     # 累積列は数百〜数千のはず（sanity）
        raise ValueError(f"{month:%Y-%m}: 開示状況列の検出が弱い（max開示済={disc_counts[status_col]}）")
    name_col = code_col + 1                                # 銘柄名は証券コードの右隣
    ind_col = None                                        # 業種コード = code より左の数値列
    for j in range(code_col):
        if df.iloc[:, j].astype(str).str.fullmatch(r"\d{2,4}").sum() >= 20:
            ind_col = j
            break
    update_col = None                                     # 開示内容のアップデート日（header 走査）
    for j in range(n):
        head = " ".join(str(df.iat[i, j]) for i in range(min(10, len(df))) if pd.notna(df.iat[i, j]))
        if "アップデート日" in head:
            update_col = j
            break

    rows = []
    for i in range(len(df)):
        sec = str(df.iat[i, code_col]).split(".")[0]
        if not re.fullmatch(r"\d{4}", sec):
            continue
        mk = str(df.iat[i, market_col]).strip()
        if mk not in ("プライム", "スタンダード"):
            continue
        status = str(df.iat[i, status_col]).strip()
        upd = df.iat[i, update_col] if update_col is not None else None
        upd = pd.to_datetime(upd, errors="coerce") if upd is not None else pd.NaT
        rows.append({
            "month": month,
            "local_code": norm_local(sec),
            "sec_code": sec,
            "market": "Prime" if mk == "プライム" else "Standard",
            "industry_code": (str(df.iat[i, ind_col]).split(".")[0] if ind_col is not None else None),
            "company_name": str(df.iat[i, name_col]).strip(),
            "disclosed": ("開示済" in status),
            "considering": ("検討中" in status),
            "status_raw": status,
            "update_date": upd,
        })
    return pd.DataFrame(rows)


def main():
    data = download()
    fetch_dt = datetime.now(JST)
    raw_path = os.path.join(RAW, f"list_{fetch_dt:%Y%m%d}.xlsx")
    with open(raw_path, "wb") as f:
        f.write(data)
    print(f"downloaded {len(data):,} bytes -> {raw_path}")

    xl = pd.ExcelFile(io.BytesIO(data))
    frames = []
    for s in xl.sheet_names:
        m = SHEET_RE.search(s)
        if not m:
            continue
        month = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) + pd.offsets.MonthEnd(0)
        raw_sheet = pd.read_excel(io.BytesIO(data), sheet_name=s, header=None)
        parsed = detect_and_parse(raw_sheet, month)
        parsed["fetched_at"] = fetch_dt.isoformat(timespec="seconds")
        frames.append(parsed)
        print(f"  {month:%Y-%m}: {len(parsed):4d} 社  (開示済 {int(parsed['disclosed'].sum())})")
    new = pd.concat(frames, ignore_index=True)

    # 冪等マージ: 既存 panel と結合し (month, local_code) 重複は最新 fetch を採用
    if os.path.exists(PANEL):
        old = pd.read_parquet(PANEL)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = (combined.sort_values("fetched_at")
                .drop_duplicates(subset=["month", "local_code"], keep="last")
                .sort_values(["month", "local_code"])
                .reset_index(drop=True))
    combined.to_parquet(PANEL, index=False)

    # 検証サマリ
    print(f"\npanel: {len(combined):,} 行 / {combined['local_code'].nunique():,} 社 / "
          f"{combined['month'].nunique()} 月次断面 ({combined['month'].min():%Y-%m}..{combined['month'].max():%Y-%m})")
    piv = (combined[combined.disclosed]
           .groupby([combined.month.dt.strftime("%Y-%m"), "market"]).size().unstack(fill_value=0))
    print("開示済 社数（月×市場）:")
    print(piv.to_string())
    # 初出（開示イベント近似）: 各社 min month で開示済
    disc = combined[combined.disclosed]
    first = disc.groupby("local_code")["month"].min()
    firstmonth = first.dt.strftime("%Y-%m").value_counts().sort_index()
    print("\n初出開示（この panel 内での初回開示済 月・左側打切り注意）:")
    print(firstmonth.to_string())
    print(f"\nsaved -> {PANEL}")


if __name__ == "__main__":
    main()
