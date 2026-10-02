"""Loader for the IQC extract (data/qc/再検データ抽出QC2023_2025.xlsx).

Raw layout: one row per QC measurement pass; the patient-ID column holds the QC material ID (one per analyzer unit x level,
constant over 2023-2025, i.e. NOT a lot number), カナ holds the material/level name, 到着時刻 the time,
then groups of (項目名, 材料, 検査値, 装置名). BM8040 passes carry 16 items, EA10M 3 (Na, K, Cl),
GA09II and HLC723G11 one item.

Material/level names (カナ) are normalised to (material, level):
  QAP L / QAP H                      -> ("QAP", "L"/"H")            main control for BM8040, EA10M, GA09II (until 2025-09)
  Multiqual 1/2, 1ﾏﾙﾁｺｰﾙ/2ﾏﾙﾁｺｰﾙ      -> ("Multiqual", "1"/"2")      GA09II from 2024-03; BM/EA short run 2024-07/08
  qap L2, QAP H2, qap H 2, ...       -> ("QAP2", "L"/"H")           parallel runs (likely new-lot evaluation)
  Diabetes / HbA1cｺﾝﾄﾛｰﾙ / 糖尿病     -> HbA1c controls (not analysed)
  機器間差 / None                     -> ("interunit", "")           inter-analyzer comparison (not QC)

build_cache() -> data/processed/iqc_results.pkl (one row per QC result; contains no patient data).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "qc" / "再検データ抽出QC2023_2025.xlsx"
CACHE = ROOT / "data" / "processed" / "iqc_results.pkl"
ITEM_NAME = {"ALP": "ALP･IFCC"}


def normalise(kana) -> tuple[str, str]:
    k = str(kana).strip()
    kl = k.lower().replace(" ", "")
    if k in ("None", "機器間差"):
        return "interunit", ""
    if "糖尿病" in k or "diabetes" in kl or "hba1c" in kl:
        return "HbA1c", k[-1]
    if "ﾏﾙﾁｺｰﾙ" in k or "multiqual" in kl:
        return "Multiqual", re.search(r"\d", k).group(0)
    if kl.startswith("qap"):
        lvl = "L" if "l" in kl[3:] else "H"
        return ("QAP2" if kl.endswith("2") else "QAP"), lvl
    return "other", k


def build_cache() -> pd.DataFrame:
    import openpyxl
    ws = openpyxl.load_workbook(RAW, read_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    next(it)
    rows = []
    for pas, r in enumerate(it):
        mat, lvl = normalise(r[3])
        for j in range(12, len(r) - 3, 4):
            if r[j] is None:
                break
            rows.append((pas, r[6], mat, lvl, r[j], r[j + 2], r[j + 3]))
    df = pd.DataFrame(rows, columns=["pass", "time", "material", "level", "item", "raw", "analyzer"])
    df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M:%S")
    df["value"] = pd.to_numeric(df["raw"], errors="coerce")
    df["unit"] = df["analyzer"].str.extract(r"#(\d+)")[0].astype("Int8")
    df["analyzer"] = df["analyzer"].str.replace(r"\s*#\d+$", "", regex=True).str.replace("Ⅱ", "II")
    for c in ("material", "level", "item", "analyzer"):
        df[c] = df[c].astype("category")
    df = df.sort_values(["time", "pass"]).reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE)
    return df


_cache = None


def results() -> pd.DataFrame:
    global _cache
    if _cache is None:
        _cache = pd.read_pickle(CACHE)
    return _cache


def load(item: str, unit: int = 1, material: str | None = None, start=None, end=None) -> pd.DataFrame:
    """Main-series QC results for one analyte on one unit (both levels), time-ordered.
    material=None: QAP, except GLU which switches to Multiqual when QAP stops (2025-09)."""
    df = results()
    m = (df["item"] == ITEM_NAME.get(item, item)) & (df["unit"] == unit) & df["value"].notna()
    if material is not None:
        m &= df["material"] == material
    else:
        m &= df["material"].isin(["QAP", "Multiqual"] if item == "GLU" else ["QAP"])
    if start is not None:
        m &= df["time"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["time"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    return df.loc[m].reset_index(drop=True)
