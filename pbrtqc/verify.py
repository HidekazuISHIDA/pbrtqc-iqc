"""Compare the result tables of a run with reference tables (verification step).

demo profile: reference = reference/demo_tables (committed; produced from the synthetic data by the same code)
real profile: reference = "reference_tables" in pbrtqc.local.json (the tables of an earlier run on the same data)

Every table in MANIFEST is compared cell by cell: text exactly, numbers with a relative tolerance of 1e-9 (same code,
same data and same random seeds give identical numbers; the tolerance only absorbs the last binary digit).
  python -m pbrtqc.verify            prints the report and writes <work>/verify_report.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C

RTOL, ATOL = 1e-9, 1e-12
_W = [f"fusion_whatif_cap{c}_qc{q}" for c in ("0.25", "0.5", "1.0") for q in ("none", "17", "17-1")]
_S = ["fusion_sens_unit1_final", "fusion_sens_unit1_final_fixed", "fusion_sens_unit2_initial", "fusion_sens_unit1_initial_delay30",
      "fusion_sens_unit1_initial_delay60", "fusion_sens_unit1_initial_bias0.25", "fusion_sens_unit1_initial_cva1.25"]
MANIFEST = [
    ("対象の記述", ["methods_flow.csv", "methods_subjects.csv", "methods_cohort.csv"]),
    ("設定の選択", ["select_dev.csv", "select_pairs.csv", "validate.csv", "select_dev_final.csv", "select_pairs_final.csv",
                "validate_final.csv", "select_pairs_noprune.csv", "prune_check.csv"]),
    ("IQC + PBRTQC", ["iqc_totalcv.csv", "fusion.csv", "fusion_trials.npz", "fusion_loo.csv", "fusion_loo_trials.npz"]
     + [f"{s}.csv" for s in _S] + [f"{w}.csv" for w in _W]),
    ("確認・補助", ["shadow_concordance.csv", "shadow_monthly.csv", "shadow_2x2.csv", "limit_widths.csv", "parvin_check.csv",
                "parvin_stages.csv", "fig1_week.csv", "fig1_limits.csv", "loo_summary.csv", "paired_ci.csv", "bias_by_sign.csv"]),
]


def _compare_csv(a: Path, b: Path) -> tuple[str, str, int, float]:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns):
        return "不一致", f"列が違う: {sorted(set(x.columns) ^ set(y.columns))[:6]}", 0, np.nan
    if len(x) != len(y):
        return "不一致", f"行数が違う: 参照 {len(x)}、今回 {len(y)}", 0, np.nan
    bad, worst = 0, 0.0
    for c in x.columns:
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            u, v = u.to_numpy(float), v.to_numpy(float)
            ok = np.isclose(u, v, rtol=RTOL, atol=ATOL, equal_nan=True)
            fin = np.isfinite(u) & np.isfinite(v)
            if fin.any():
                worst = max(worst, float(np.max(np.abs(u[fin] - v[fin]) / np.maximum(np.abs(u[fin]), 1e-300))))
        else:
            na_u, na_v = u.isna().to_numpy(), v.isna().to_numpy()          # empty cells are equal to each other
            ok = (na_u & na_v) | (~na_u & ~na_v & (u.astype(object).astype(str) == v.astype(object).astype(str)).to_numpy())
        bad += int((~ok).sum())
    return ("一致" if bad == 0 else "不一致"), (f"{bad} セルが違う" if bad else ""), len(x), worst


def _compare_npz(a: Path, b: Path) -> tuple[str, str, int, float]:
    x, y = np.load(a), np.load(b)
    if sorted(x.files) != sorted(y.files):
        return "不一致", "配列の名前が違う", 0, np.nan
    bad = sum(int((~np.isclose(x[k], y[k], rtol=RTOL, atol=ATOL, equal_nan=True)).sum()) for k in x.files)
    return ("一致" if bad == 0 else "不一致"), (f"{bad} 値が違う" if bad else ""), len(x.files), np.nan


def compare(reference: Path | None = None, current: Path | None = None) -> pd.DataFrame:
    reference = Path(reference or C.REFERENCE or "")
    current = Path(current or C.TABLES)
    if not reference.is_dir():
        raise FileNotFoundError(f"reference tables not found: {reference} (set 'reference_tables' in pbrtqc.local.json)")
    rows = []
    for group, names in MANIFEST:
        for n in names:
            a, b = reference / n, current / n
            if not a.exists():
                rows.append((group, n, "参照なし", "参照側にファイルがない", 0, np.nan))
            elif not b.exists():
                rows.append((group, n, "未作成", "今回の実行で作られていない", 0, np.nan))
            else:
                rows.append((group, n, *(_compare_npz(a, b) if n.endswith(".npz") else _compare_csv(a, b))))
    rep = pd.DataFrame(rows, columns=["区分", "ファイル", "判定", "内容", "行数", "最大相対差"])
    rep.to_csv(C.WORK / "verify_report.csv", index=False)
    return rep


def main():
    print(C.describe())
    rep = compare()
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))
    n_ok = (rep["判定"] == "一致").sum()
    print(f"\n{n_ok} / {len(rep)} 件が一致")
    sys.exit(0 if n_ok == len(rep) else 1)


if __name__ == "__main__":
    main()
