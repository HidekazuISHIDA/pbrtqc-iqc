"""Synthetic demonstration data for the monitoring screen (no real patient data).

Simulates one analyzer for the 20 study analytes (same order as the paper): weekday case mix (inpatients early morning, outpatients during the day),
patient set points (between-subject variation, lower values in inpatients where typical), repeat visits
(daily draws for inpatients, occasional visits for outpatients), within-subject and analytical variation, and two
QC levels twice a day. A systematic error of +1.5 x TEa on Ca is injected from 2025-01-20 08:00 to
2025-01-23 10:00 into both patient results and QC. Development (limits) = Mar-Dec 2024, N = 300 (false alarms about 0.25/week/analyte, as in the study); replay = January 2025.

The statistics are computed with the same code as the study (pbrtqc.realsim): average of patient deltas (AoD),
even check and an outpatient moving average, plus QC scores. Output: demo/data.json (same schema as the real
monitor) with "demo": true so the page shows a synthetic-data banner.

Usage: python3 scripts/demo_make_data.py [output_dir]   (default outputs/monitor_demo)
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import realsim as R

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/monitor_demo"
rng = np.random.default_rng(2025)
import os
NW = int(os.environ.get("DEMO_N", 300))
START, DEV_END, END = pd.Timestamp(os.environ.get("DEMO_DEV0", "2024-03-01")), pd.Timestamp("2024-12-31 23:59"), pd.Timestamp("2025-01-31 23:00")
R.DEV = (str(START.date()), "2024-12-31")
ERR = dict(item="Ca", start=pd.Timestamp("2025-01-20 08:00"), end=pd.Timestamp("2025-01-23 10:00"), k=1.5)
# name: median, between-subject CV, within-subject CV, analytical CV, TEa, inpatient shift, decimals, QC L/H conc.
# TEa = JSCC 2006 desirable (BA + 1.65 x CVA; Mg: EFLM). Other values are generic illustrative numbers, not the study's.
A = {
    "Na":  (140, .015, .006, .006, .0096, 0.00, 0, (130, 150)),
    "K":   (4.2, .08, .05, .007, .062, -0.03, 1, (4.0, 6.0)),
    "Cl":  (104, .02, .012, .006, .0165, -0.01, 0, (95, 115)),
    "Ca":  (9.0, .04, .02, .012, .0315, -0.05, 1, (7.5, 10.5)),
    "TP":  (6.8, .07, .03, .007, .0368, -0.12, 1, (5.0, 7.5)),
    "Alb": (4.0, .10, .03, .015, .0394, -0.20, 1, (3.0, 4.5)),
    "BUN": (14, .30, .12, .020, .1771, 0.15, 1, (15, 50)),
    "Cre": (0.8, .25, .06, .008, .0925, 0.00, 2, (1.0, 4.4)),
    "UA":  (5.3, .25, .08, .015, .1376, -0.10, 1, (3.5, 8.0)),
    "AST": (21, .35, .12, .010, .196, 0.05, 0, (35, 115)),
    "ALT": (18, .50, .15, .020, .3071, 0.05, 0, (30, 120)),
    "LD":  (190, .18, .07, .015, .0951, 0.05, 0, (150, 400)),
    "ALP": (80, .30, .07, .020, .1293, 0.10, 0, (100, 350)),
    "GGT": (25, .70, .12, .020, .2633, 0.10, 0, (40, 150)),
    "CK":  (100, .55, .25, .020, .2961, -0.20, 0, (100, 400)),
    "AMY": (75, .30, .09, .015, .1373, -0.05, 0, (60, 300)),
    "TG":  (110, .50, .20, .020, .3982, 0.00, 0, (80, 200)),
    "TC":  (190, .18, .06, .012, .1011, -0.20, 0, (150, 250)),
    "Mg":  (2.0, .08, .04, .016, .0385, 0.00, 1, (1.5, 4.5)),
    "Glu": (108, .15, .10, .010, .0708, -0.03, 0, (80, 250)),
}

# ---- arrivals and patients -----------------------------------------------------------------
days = pd.date_range(START, END.normalize())
rows, pid = [], 0
inpatients = []                                    # [pid, discharge day]
outpatients = list(range(-1, -6001, -1))           # pool of outpatient ids (negative), revisit at random
for day in days:
    wk = day.dayofweek < 5
    inpatients = [p for p in inpatients if p[1] > day]
    while len(inpatients) < 220:
        pid += 1
        inpatients.append([pid, day + pd.Timedelta(days=int(rng.integers(3, 21)))])
    for p, _ in inpatients:                         # daily morning draw for ~70 % of inpatients
        if rng.random() < (0.7 if wk else 0.35):
            rows.append((p, True, day + pd.Timedelta(hours=float(rng.uniform(5, 8)))))
    n_out = int(rng.poisson(200 if wk else 8))
    for p in rng.choice(outpatients, n_out, replace=False):
        rows.append((int(p), False, day + pd.Timedelta(hours=float(np.clip(rng.normal(10.5, 1.6), 8.3, 16.5)))))
    for _ in range(int(rng.poisson(20))):           # evening / emergency
        pid += 1
        rows.append((pid, True, day + pd.Timedelta(hours=float(rng.uniform(16, 23.9)))))
pat = pd.DataFrame(rows, columns=["pkey", "inpat", "arrival"]).sort_values("arrival").reset_index(drop=True)
pat["acc"] = np.arange(len(pat))
pat["sid"] = pat["acc"]
pat["date"] = pat["arrival"].dt.normalize()
pat["unit"] = 1
pat["sex"] = np.where(pat["pkey"] % 2 == 0, "男性", "女性")
setpt = {}

hours = pd.date_range("2025-01-01 00:00", END, freq="h")
ends = (hours + pd.Timedelta(hours=1)).to_numpy()


def hourly(t, v):
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    i = np.searchsorted(t, ends, "right") - 1
    return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)


def rnd(a):
    return [None if not np.isfinite(x) else round(float(x), 2) for x in a]


qc_times = []
for day in days:
    qc_times += [day + pd.Timedelta(hours=4.5)] + ([day + pd.Timedelta(hours=10.5)] if day.dayofweek < 5 else [])
qc_t = np.array(qc_times, dtype="datetime64[ns]")

items = []
for name, (med, cvg, cvi, cva, tea, shift, dec, (qL, qH)) in A.items():
    d = pat.copy()
    keys = d["pkey"].unique()
    sp = dict(zip(keys, med * np.exp(rng.normal(0, cvg, keys.size))))
    base = d["pkey"].map(sp).to_numpy() * np.where(d["inpat"], 1 + shift, 1.0)
    v = base * (1 + rng.normal(0, cvi, len(d))) * (1 + rng.normal(0, cva, len(d)))
    if name == ERR["item"]:
        hit = ((d["arrival"] >= ERR["start"]) & (d["arrival"] < ERR["end"])).to_numpy()
        v[hit] *= 1 + ERR["k"] * tea
    d["value"] = np.round(v, dec)
    meta = dict(item=name, key=name, unit="", analyzer="Demo analyzer", tea=round(tea * 100, 2), cva=round(cva * 100, 2),
                sigma=round(tea / cva, 1), setting="demo", comps={}, demo=True)
    scores, episodes = {}, []
    comps = {"aod": R.make_aod_stream(R.delta_table(d, lookback=30), "2.5-97.5", NW),
             "even": R.make_aod_stream(R.delta_table(d, lookback=30), "none", NW, kind="even"),
             "out": R.make_stream(d, R.stream_masks(d)["out"], name, "2.5-97.5", NW)}
    settings = {"aod": f"2.5-97.5|G30/N{NW}", "even": f"G30/N{NW}", "out": f"2.5-97.5/N{NW}"}
    for k, st in comps.items():
        mid, half = (st.cl[0] + st.cl[1]) / 2, (st.cl[1] - st.cl[0]) / 2
        scores[k] = hourly(st.t, (st.stat - mid) / half)
        meta["comps"][k] = dict(setting=settings[k], centre=float(mid), half=float(half))
        for s_, e_ in R.episodes(st):
            if s_ >= np.datetime64("2025-01-01"):
                episodes.append(dict(c=k, s=str(s_)[:16], e=str(e_)[:16]))
    for lvl, conc in (("qcL", qL), ("qcH", qH)):
        x = conc * (1 + rng.normal(0, cva, qc_t.size))
        if name == ERR["item"]:
            x[(qc_t >= np.datetime64(ERR["start"])) & (qc_t < np.datetime64(ERR["end"]))] *= 1 + ERR["k"] * tea
        daily = pd.Series(x, index=pd.DatetimeIndex(qc_t)).groupby(pd.DatetimeIndex(qc_t).normalize()).median()
        cen = daily.shift(1).rolling(30, min_periods=10).median().reindex(pd.DatetimeIndex(qc_t).normalize()).to_numpy()
        z = (x - cen) / (cva * cen)
        scores[lvl] = hourly(qc_t, z / 3)
        meta["comps"][lvl] = dict(setting="QC", centre=float(cen[-1]), half=float(3 * cva * cen[-1]))
    meta["scores"] = {k: rnd(scores[k]) for k in ["qcL", "qcH", "aod", "even", "out"]}
    meta["episodes"] = sorted(episodes, key=lambda e: e["s"])
    items.append(meta)
    print(name, len(episodes), "episodes", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
data = dict(hours=[hours[0].isoformat(), len(hours)], comps=["qcL", "qcH", "aod", "even", "out"], items=items, demo=True,
            demo_note=f"Injected error: {ERR['item']} +{ERR['k']} x TEa from {ERR['start']:%Y-%m-%d %H:%M} to {ERR['end']:%Y-%m-%d %H:%M}.")
(OUT / "data.json").write_text(json.dumps(data, separators=(",", ":")))
print("written", OUT / "data.json", (OUT / "data.json").stat().st_size // 1024, "KB")
