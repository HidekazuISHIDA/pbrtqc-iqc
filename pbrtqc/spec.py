"""Study specification for the 21 analytes — the single place where the design decisions live.

Sources
  TEa            docs/10_TEa定義.md  (JSCC 2006, TEa = BA + 1.65 x CVA; no Na/Cl exception, no 5% cap)
  CVa            docs/09_項目選定.md (自施設 2024 測定不確かさ推定記録の合成標準不確かさ u_c, L 側)
                 ** conservative: u_c includes the calibrator's uncertainty. See docs/10 §8. **
  distribution   outputs/tables/adult_dev_2023_2024.csv (scripts/03_adult_dev.py: 設定期間 2023-2024、18 歳以上。
                 2023 年以降の旧抽出はもともと 18 歳以上のみ)
  analyzers      docs/09_項目選定.md
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# item: (JSCC CVA%, JSCC BA%)  -> TEa% = BA + 1.65*CVA
JSCC = {
    "Na": (0.4, 0.3), "K": (2.6, 1.9), "Cl": (0.7, 0.5), "Ca": (1.3, 1.0), "TP": (1.5, 1.2),
    "ALB": (1.6, 1.3), "UN": (7.1, 6.0), "CRE": (2.7, 4.8), "UA": (4.4, 6.5), "AST": (7.6, 7.1),
    "ALT": (11.1, 12.4), "LD": (3.4, 3.9), "ALP": (3.9, 6.5), "γ-GT": (8.2, 12.8),
    "CK": (11.1, 11.3), "AMY": (4.2, 6.8), "GLU": (2.9, 2.3), "TG": (14.8, 15.4), "T-CHO": (3.4, 4.5),
}
# 自施設 u_c (%) — 合成標準不確かさ, L 側 (BM8040#1 / EA10M#1 / GA09II#1)
UC = {
    "Na": 0.835, "K": 0.612, "Cl": 0.874, "Ca": 1.277, "TP": 1.481, "ALB": 2.089, "UN": 1.041,
    "CRE": 0.960, "UA": 1.240, "AST": 1.724, "ALT": 1.946, "LD": 1.440, "ALP": 2.436,
    "γ-GT": 3.165, "CK": 1.859, "AMY": 1.399, "GLU": 1.382, "TG": 1.810, "T-CHO": 0.910,
    "Mg": 2.834, "HbA1c": 1.633,
}
ANALYZERS = {  # 装置と台数
    "Na": ("EA10M", 2), "K": ("EA10M", 2), "Cl": ("EA10M", 2), "GLU": ("GA09II", 2),
    "HbA1c": ("HLC723G11", 2),
}
DEFAULT_ANALYZER = ("BM8040", 2)
# 基準範囲（打ち切り候補 "ref" 用）: 自施設は JCCLS 共用基準範囲を使用（ユーザー確認 2026-09-29）。
# 出典: 日本臨床検査標準協議会 基準範囲共用化委員会「日本における主要な臨床検査項目の共用基準範囲
#       ―解説と利用の手引き―」2022/10/01 版, 表 1-1。値は ~/Downloads/kijyunhani20221031.pdf から転記。
# 男女別の項目は {"M": (下限, 上限), "F": (...)}。Mg は共用基準範囲に無く、自施設設定 1.8–2.4 mg/dL（ユーザー 2026-09-29）。
REF_JCCLS = {
    "Na": (138, 145), "K": (3.6, 4.8), "Cl": (101, 108), "Ca": (8.8, 10.1), "TP": (6.6, 8.1),
    "ALB": (4.1, 5.1), "UN": (8, 20), "CRE": {"M": (0.65, 1.07), "F": (0.46, 0.79)},
    "UA": {"M": (3.7, 7.8), "F": (2.6, 5.5)}, "AST": (13, 30), "ALT": {"M": (10, 42), "F": (7, 23)},
    "LD": (124, 222), "ALP": (38, 113), "γ-GT": {"M": (13, 64), "F": (9, 32)},
    "CK": {"M": (59, 248), "F": (41, 153)}, "AMY": (44, 132), "GLU": (73, 109),
    "TG": {"M": (40, 234), "F": (30, 117)}, "T-CHO": (142, 248), "Mg": (1.8, 2.4), "HbA1c": (4.9, 6.0),
}
# 旧: 性別をまとめた暫定値（scripts/40・50・51 が使用）
REF = {
    "Na": (138, 145), "K": (3.6, 4.8), "Cl": (101, 108), "Ca": (8.8, 10.1), "TP": (6.6, 8.1),
    "ALB": (4.1, 5.1), "UN": (8, 20), "CRE": (0.46, 1.07), "UA": (3.7, 7.8), "AST": (13, 30),
    "ALT": (10, 42), "LD": (124, 222), "ALP": (38, 113), "γ-GT": (13, 64), "CK": (59, 248),
    "AMY": (44, 132), "GLU": (73, 109), "TG": (40, 234), "T-CHO": (142, 248),
    "Mg": (1.8, 2.4), "HbA1c": (4.9, 6.0),
}


def ref_limits(item: str, sex) -> tuple | None:
    """Per-result (L, U) arrays from the JCCLS common reference interval; None if undefined.
    sex: array of '男性' / '女性'."""
    r = REF_JCCLS.get(item)
    if r is None:
        return None
    sex = np.asarray(sex)
    if isinstance(r, tuple):
        return (np.full(sex.size, float(r[0])), np.full(sex.size, float(r[1])))
    m = sex == "男性"
    return (np.where(m, r["M"][0], r["F"][0]).astype(float), np.where(m, r["M"][1], r["F"][1]).astype(float))


DECIMALS = {"K": 1, "CRE": 2, "Ca": 1, "TP": 1, "ALB": 1, "Mg": 1, "HbA1c": 1, "TG": 0, "UN": 1}
UNITS = {
    "Na": "mmol/L", "K": "mmol/L", "Cl": "mmol/L", "Ca": "mg/dL", "TP": "g/dL", "ALB": "g/dL",
    "UN": "mg/dL", "CRE": "mg/dL", "UA": "mg/dL", "AST": "U/L", "ALT": "U/L", "LD": "U/L",
    "ALP": "U/L", "γ-GT": "U/L", "CK": "U/L", "AMY": "U/L", "GLU": "mg/dL", "TG": "mg/dL",
    "T-CHO": "mg/dL", "Mg": "mg/dL", "HbA1c": "%",
}
ITEMS21 = ["Na", "K", "Cl", "Ca", "TP", "ALB", "UN", "CRE", "UA", "AST", "ALT", "LD", "ALP",
           "γ-GT", "CK", "AMY", "GLU", "TG", "T-CHO", "Mg", "HbA1c"]
# 解析対象: HbA1c を除く 20 項目（2026-09-28 決定、docs/14 §2.2）。件数の足切りはしない。
EXCLUDED = {"HbA1c": "件数少・JSCC に TEa なし・HPLC/% 単位・糖尿病経過観察に偏る"}
ITEMS = [i for i in ITEMS21 if i not in EXCLUDED]


def load(dist_csv=None) -> dict:
    """Return {item: spec dict} for the 21 analytes (development period 2023-2024, adults)."""
    path = Path(dist_csv or ROOT / "outputs" / "tables" / "adult_dev_2023_2024.csv")
    with open(path, encoding="utf-8") as f:
        dist = {r["item"]: r for r in csv.DictReader(f)}
    out = {}
    for it in ITEMS:
        d = dist.get(it)
        if d is None:
            continue
        analyzer, n_an = ANALYZERS.get(it, DEFAULT_ANALYZER)
        per_day = float(d["per_day_median"])
        cva = UC[it] / 100.0
        if it in JSCC:
            cvA, bA = JSCC[it]
            tea = (bA + 1.65 * cvA) / 100.0
            tea_src = "JSCC2006"
        else:
            tea, tea_src = None, "未定義"
        out[it] = dict(
            item=it, unit=UNITS[it], analyzer=analyzer, n_analyzers=n_an,
            per_day_total=per_day, per_day_per_analyzer=per_day / n_an,
            weekday_per_analyzer=float(d["weekday_median"]) / n_an,
            weekend_per_analyzer=float(d["weekend_median"]) / n_an,
            inpat_frac=float(d["inpat_frac"]),
            inpat_frac_weekday=float(d["inpat_frac_weekday"]),
            inpat_frac_weekend=float(d["inpat_frac_weekend"]),
            p2_5=float(d["p2_5"]), median=float(d["median"]), p97_5=float(d["p97_5"]),
            cva=cva, tea=tea, tea_source=tea_src,
            sigma=(tea / cva) if tea else None,
            ref=REF.get(it), decimals=DECIMALS.get(it, 0),
        )
    return out


def lognormal_params(med, lo, hi):
    """Fit a lognormal to median and the 2.5/97.5 percentiles; returns (mu, sigma_log, skewness flag)."""
    med = max(med, 1e-9)
    s_hi = np.log(max(hi, med * 1.0001) / med) / 1.959964
    s_lo = np.log(med / max(lo, med * 1e-4)) / 1.959964
    return np.log(med), float((s_hi + s_lo) / 2), float(s_hi / max(s_lo, 1e-9))


def normal_params(med, lo, hi):
    return med, float((hi - lo) / (2 * 1.959964))


def main_items(spec=None):
    """All analysed items (no workload threshold). Whether PBRTQC 'works' is decided from results."""
    spec = spec or load()
    return list(spec)


if __name__ == "__main__":
    s = load()
    print(f"{'item':<8}{'装置':<11}{'台':>3}{'件/日':>7}{'件/日/台':>9}{'CVa%':>7}{'TEa%':>7}{'σ':>6}")
    for k, v in s.items():
        print(f"{k:<8}{v['analyzer']:<11}{v['n_analyzers']:>3}{v['per_day_total']:>7.0f}"
              f"{v['per_day_per_analyzer']:>9.0f}{v['cva']*100:>7.2f}"
              f"{(v['tea']*100 if v['tea'] else float('nan')):>7.2f}"
              f"{(v['sigma'] if v['sigma'] else float('nan')):>6.1f}")
    print("\n解析対象:", len(main_items(s)), "項目 / 除外:", ", ".join(EXCLUDED))


# --- names for the manuscript (Clinica Chimica Acta), 2026-10-01 ------------------------------------
# Internal keys (LIS-derived) stay unchanged in code and files; DISPLAY is used for tables, legends and the monitor.
DISPLAY = {"ALB": "Alb", "UN": "BUN", "CRE": "Cre", "γ-GT": "GGT", "T-CHO": "TC", "GLU": "Glu"}
FULL_NAME = {"Na": "sodium", "K": "potassium", "Cl": "chloride", "Ca": "calcium", "Mg": "magnesium",
             "TP": "total protein", "ALB": "albumin", "UN": "blood urea nitrogen", "CRE": "creatinine",
             "UA": "uric acid", "AST": "aspartate aminotransferase", "ALT": "alanine aminotransferase",
             "LD": "lactate dehydrogenase", "ALP": "alkaline phosphatase", "γ-GT": "γ-glutamyltransferase",
             "CK": "creatine kinase", "AMY": "amylase", "GLU": "glucose", "TG": "triglycerides",
             "T-CHO": "total cholesterol"}


def disp(item: str) -> str:
    return DISPLAY.get(item, item)
