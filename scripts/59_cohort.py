"""Cohort description for the manuscript (analyzer #1, adults, initial numeric results of the 20 analytes, 2023-2025).

Counts distinct pseudonymous patients (pkey) and specimens (sid) across the 20 analytes; patient-level age and sex
summarised at each patient's first specimen and at specimen level; inpatient share over results. Output (aggregates only):
<tables>/methods_cohort.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import config as C, lis, spec

parts = []
for it in spec.ITEMS:
    d = lis.load(it, unit=1, start="2023-01-01", end="2025-12-31")
    d = d[d["value"].notna()]
    parts.append(d[["pkey", "sid", "arrival", "inpat", "sex", "age"]])
a = pd.concat(parts, ignore_index=True)
first = a.sort_values("arrival").drop_duplicates("pkey")
q = first["age"].quantile([.25, .5, .75])
out = dict(results=len(a), patients=a["pkey"].nunique(), specimens=a["sid"].nunique(),
           age_median=float(q[.5]), age_q1=float(q[.25]), age_q3=float(q[.75]),
           male_pct=float((first["sex"] == "男性").mean() * 100), inpatient_results_pct=float(a["inpat"].mean() * 100))
s = a.drop_duplicates("sid")                       # specimen level (earlier Methods draft used this level)
qs = s["age"].quantile([.25, .5, .75])
out.update(spec_age_median=float(qs[.5]), spec_age_q1=float(qs[.25]), spec_age_q3=float(qs[.75]),
           spec_male_pct=float((s["sex"] == "男性").mean() * 100), spec_inpatient_pct=float(s["inpat"].mean() * 100))
pd.DataFrame([out]).to_csv(C.TABLES / "methods_cohort.csv", index=False)
print(out)
