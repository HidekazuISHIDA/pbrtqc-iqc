"""Loader for the 2026-09 LIS extract (data/raw/lis_2023_2025/*.xlsx, patient results only).

Raw layout: one row per measurement pass of a specimen; fixed columns 受付日, <patient ID column>,
依頼元, 到着時刻, HISDATE (specimen ID), 年齢, 入外区分, 依頼科名, 性別, then up to 24 groups of
(項目名, 材料, 検査値, 装置名).

Re-measurements: ~19 % of specimens have 2+ rows, always adjacent, same patient and arrival time.
The first row is the initial run and later rows are reruns (「要希釈再検」 appears almost only in
the first row; about half of reruns are on the other analyzer unit). For each specimen x item,
`rep` numbers the rows in file order, `final` marks the last row carrying that item (the reported
value) and `initial` the first.

build_cache() flattens this to one row per result and writes data/processed/lis_results.pkl.
The patient ID and the specimen ID are replaced by sequential keys `pkey` / `sid` (the patient key is
needed to link a patient's previous result for even check); the original IDs are not written anywhere.

load(item, unit) returns the analysis stream for one analyte on one analyzer: adults (>=18 y),
the basic specimen (serum; GLU whole blood in NaF tube = 材料「血液」), numeric results only,
one value per specimen, ordered by arrival time. value="initial" (default, primary analysis: the
first measurement, i.e. what the analyzer produced at that time) or "final" (reported value, sensitivity).
The analyzer unit is the one that produced the selected value.
"""
from __future__ import annotations
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lis_2023_2025"
CACHE = ROOT / "data" / "processed" / "lis_results.pkl"

ITEM_NAME = {"ALP": "ALP･IFCC"}                    # analysis name -> LIS name
MATERIAL = {"GLU": "血液", "HbA1c": "血液"}        # default 血清
SERUM_INDEX = ("検体溶血", "検体乳ビ", "検体黄疸")
MIN_AGE = 18


def age_years(s: str) -> float:
    """'76歳', '1歳3ヶ月', '5ヶ月12日', '9日' -> years."""
    s = str(s)
    y = re.search(r"(\d+)歳", s); m = re.search(r"(\d+)ヶ月", s); d = re.search(r"(\d+)日", s)
    if not (y or m or d):
        return np.nan
    return (int(y.group(1)) if y else 0) + (int(m.group(1)) / 12 if m else 0) + (int(d.group(1)) / 365.25 if d else 0)


def to_value(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Numeric value and censoring flag ('<', '>' or '')."""
    s = raw.astype(str).str.strip()
    flag = s.str.extract(r"^([<>])")[0].fillna("")
    val = pd.to_numeric(s.str.replace(r"^[<>]", "", regex=True).str.replace(",", ""), errors="coerce")
    return val, flag


def build_cache(files=None) -> pd.DataFrame:
    import openpyxl
    files = sorted(files or glob.glob(str(RAW / "*.xlsx")))
    acc_rows, res_rows = [], []
    pid_map: dict[str, int] = {}
    sid_map: dict[str, int] = {}
    acc_id = 0
    for f in files:
        ws = openpyxl.load_workbook(f, read_only=True).worksheets[0]
        it = ws.iter_rows(values_only=True)
        next(it)
        for r in it:
            acc_id += 1
            pk = pid_map.setdefault(str(r[1]), len(pid_map) + 1)
            sk = sid_map.setdefault(str(r[4]), len(sid_map) + 1)
            acc_rows.append((acc_id, sk, r[0], r[3], pk, r[2], r[6], r[7], r[8], r[5]))
            for j in range(9, len(r) - 3, 4):
                if r[j] is None:
                    break
                res_rows.append((acc_id, r[j], r[j + 1], r[j + 2], r[j + 3]))
    acc = pd.DataFrame(acc_rows, columns=["acc", "sid", "date", "arrival", "pkey", "source", "inout",
                                          "dept", "sex", "age_raw"])
    res = pd.DataFrame(res_rows, columns=["acc", "item", "material", "raw", "analyzer"])
    del acc_rows, res_rows, pid_map, sid_map
    acc["rep"] = acc.groupby("sid").cumcount().astype("int8")      # file order = measurement order
    acc["date"] = pd.to_datetime(acc["date"], format="%Y/%m/%d")
    acc["arrival"] = pd.to_datetime(acc["arrival"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    acc["age"] = acc.pop("age_raw").map(age_years).astype("float32")
    acc["inpat"] = acc["inout"].eq("入院")
    # serum indices as accession-level columns (hemolysis / lipemia / icterus)
    idx = res[res["item"].isin(SERUM_INDEX)].pivot_table(index="acc", columns="item", values="raw",
                                                           aggfunc="first")
    acc = acc.join(idx, on="acc")
    res = res[~res["item"].isin(SERUM_INDEX)].copy()
    res["value"], res["flag"] = to_value(res["raw"])
    res["unit"] = res["analyzer"].str.extract(r"#(\d+)")[0].astype("Int8")
    res["analyzer"] = res["analyzer"].str.replace(r"\s*#\d+$", "", regex=True).str.replace("Ⅱ", "II")
    for c in ("item", "material", "analyzer", "flag"):
        res[c] = res[c].astype("category")
    for c in ("source", "inout", "dept", "sex") + SERUM_INDEX:
        acc[c] = acc[c].astype("category")
    out = res.merge(acc, on="acc", how="left")
    g = out.groupby(["sid", "item"], observed=True)["rep"]
    out["final"] = out["rep"].eq(g.transform("max"))
    out["initial"] = out["rep"].eq(g.transform("min"))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(CACHE)
    return out


_cache: pd.DataFrame | None = None


def results() -> pd.DataFrame:
    global _cache
    if _cache is None:
        _cache = pd.read_pickle(CACHE)
    return _cache


VALUE = "initial"   # module default; scripts may set lis.VALUE = "final" for the sensitivity analysis


def load(item: str, unit: int | None = 1, adult=True, start=None, end=None,
         value: str | None = None) -> pd.DataFrame:
    """Analysis stream for one analyte (and analyzer unit), one value per specimen, by arrival time."""
    df = results()
    m = (df["item"] == ITEM_NAME.get(item, item)) & (df["material"] == MATERIAL.get(item, "血清"))
    m &= df[value or VALUE]
    m &= df["value"].notna()
    if unit is not None:
        m &= df["unit"] == unit
    if adult:
        m &= df["age"] >= MIN_AGE
    if start is not None:
        m &= df["date"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["date"] <= pd.Timestamp(end)
    cols = ["acc", "sid", "rep", "date", "arrival", "pkey", "inpat", "source", "dept", "sex", "age",
            "value", "flag", "analyzer", "unit"] + list(SERUM_INDEX)
    return df.loc[m, cols].sort_values(["arrival", "acc"]).reset_index(drop=True)
