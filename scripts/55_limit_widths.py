"""Control-limit half-widths by monitoring approach (Figure 2B).

Analyzer #1, initial values, development period 2023-24, N = 50, truncation 2.5-97.5 percentile.
Half-width = 3 SD of the statistic, expressed as % of the median result of the item, and as a ratio to TEa
(< 1 means a 1 x TEa shift moves the statistic beyond the limit).
Approaches: pooled MA, inpatient MA, outpatient MA, AoD (lookback 7 d and 90 d).
Output: outputs/tables/limit_widths.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R

N, TR = 50, "2.5-97.5"
S = spec.load()
rows = []
for it in spec.ITEMS:
    tea = R.tea(it, S) * 100
    d = lis.load(it, unit=1, start=R.DEV[0], end=R.VAL[1])
    med = d.loc[d["date"] <= R.DEV[1], "value"].median()
    r = dict(item=it, tea_pct=tea, median_result=med)
    for name, mask in R.stream_masks(d).items():
        st = R.make_stream(d, mask, it, TR, N)
        r[f"{name}_MA"] = 3 * np.nanstd(st.stat[st.dev]) / med * 100
    d_all = lis.load(it, unit=None, start=R.DEV[0], end=R.VAL[1])
    for G in (7, 90):
        st = R.make_aod_stream(R.delta_table(d_all, lookback=G), TR, N)
        r[f"AoD_G{G}"] = 3 * np.nanstd(st.stat[st.dev]) / med * 100
        r[f"AoD_G{G}_per_day"] = int(st.dev.sum() / 731)
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs/tables/limit_widths.csv", index=False)
print(out.round(2).to_string(index=False))
