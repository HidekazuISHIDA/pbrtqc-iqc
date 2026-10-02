"""Figure 1 data: Alb on analyzer #1 over a representative weekday week (Mon-Fri, 2024-06-10..14, development period).

Three statistics with the Figure 2 settings (N = 50, truncation 2.5-97.5 %, AoD lookback 90 d):
pooled MA, outpatient MA and AoD. Each is expressed as its deviation from its own centre (mean of the
statistic in 2023-2024) in % of the median Alb result, so the three share one axis. Also: each statistic's
control limit (3 SD, same units) and TEa. Value shown for an hour = last value at the end of that hour.
Output: outputs/tables/fig1_week.csv, outputs/tables/fig1_limits.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R

ITEM, N, TR, G = "ALB", 50, "2.5-97.5", 90
START, END = pd.Timestamp("2024-06-10 00:00"), pd.Timestamp("2024-06-14 23:00")
assert START.dayofweek == 0
S = spec.load()
d = lis.load(ITEM, unit=1, start=R.DEV[0], end=R.VAL[1])
med = d.loc[d["date"] <= R.DEV[1], "value"].median()
hours = pd.date_range(START, END, freq="h")
ends = (hours + pd.Timedelta(hours=1)).to_numpy()


def hourly(t, v):
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    i = np.searchsorted(t, ends, "right") - 1
    return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)


out = pd.DataFrame({"Time": hours, "Day": hours.strftime("%a"), "Hour": hours.hour})
lim = []
streams = {"Pooled MA": R.make_stream(d, R.stream_masks(d)["pooled"], ITEM, TR, N),
           "Outpatient MA": R.make_stream(d, R.stream_masks(d)["out"], ITEM, TR, N),
           "AoD": R.make_aod_stream(R.delta_table(lis.load(ITEM, unit=None, start=R.DEV[0], end=R.VAL[1]), lookback=G), TR, N)}
for name, st in streams.items():
    centre, half = (st.cl[0] + st.cl[1]) / 2, (st.cl[1] - st.cl[0]) / 2
    out[f"{name} (% of median Alb)"] = (hourly(st.t, st.stat) - centre) / med * 100
    lim.append(dict(Statistic=name, centre=centre, half_width=half, **{"Control limit (± % of median Alb)": half / med * 100}))
tea = R.tea(ITEM, S) * 100
lim.append(dict(Statistic="TEa", **{"Control limit (± % of median Alb)": tea}))
out.to_csv(ROOT / "outputs/tables/fig1_week.csv", index=False)
pd.DataFrame(lim).to_csv(ROOT / "outputs/tables/fig1_limits.csv", index=False)
print(pd.DataFrame(lim).round(3).to_string(index=False))
print(out.iloc[:, 3:].describe().round(2).loc[["min", "max"]].to_string())
