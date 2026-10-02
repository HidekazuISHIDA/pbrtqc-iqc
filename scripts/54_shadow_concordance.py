"""Shadow operation on the real data: do PBRTQC alarms coincide with real laboratory actions? (docs/16 layer 2)

No error is injected. Primary PBRTQC system = 3-way (AoD + even check + outpatient MA), settings chosen on
2023-24 under the 0.25/week cap (select_pairs.csv), run over 2023-2025 on analyzer #1.

Laboratory actions: "extra" QC events of the analyzer (not the first morning / midday QC; after calibration,
reagent change or a rule violation, user 2026-09-30). The real QC values are too irregular (level swaps,
outliers, drift, no lot numbers) to replay the Westgard decisions, so extra QC is used as the event proxy
(docs/14 §6.3).

Concordance: a PBRTQC alarm episode is "matched" if an extra QC event lies within +/-12 h of it. Chance level
= share of patient-result times on the analyzer that lie within 12 h of an extra QC event; enrichment =
matched share / chance. Also monthly alarm counts and the monthly deviation of the QC medians (L, H) from
their 2023-24 median, to see whether the higher 2025 alarm rate follows analytical shifts.

Outputs (aggregates only): outputs/tables/shadow_concordance.csv, outputs/tables/shadow_monthly.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R, iqc_model as M, iqc_data as Q

WIN = np.timedelta64(12, "h")
S = spec.load()
pairs = pd.read_csv(ROOT / "outputs/tables/select_pairs.csv")
conc, monthly = [], []


def build(key, d, dtabs, item):
    kind, rest = key.split(":")
    trunc, N = rest.rsplit("/N", 1)
    N = int(N)
    if kind in ("in", "out", "pooled"):
        return R.make_stream(d, R.stream_masks(d)[kind], item, trunc, N)
    if kind == "aod":
        tr, G = trunc.split("|G")
        return R.make_aod_stream(dtabs[int(G)], tr, N)
    return R.make_aod_stream(dtabs[int(trunc[1:])], "none", N, kind="even")


def near(times, ref, win=WIN):
    """For each time, is there a ref time within +/-win?"""
    ref = np.sort(ref)
    i = np.searchsorted(ref, times)
    lo = np.abs(times - ref[np.clip(i - 1, 0, ref.size - 1)]) <= win
    hi = np.abs(ref[np.clip(i, 0, ref.size - 1)] - times) <= win
    return lo | hi


for it in spec.ITEMS:
    d = lis.load(it, unit=1, start=R.DEV[0], end=R.VAL[1])
    d_all = lis.load(it, unit=None, start=R.DEV[0], end=R.VAL[1])
    dtabs = {G: R.delta_table(d_all, lookback=G) for G in R.AOD_LOOKBACK}
    setting = pairs[(pairs["item"] == it) & (pairs["cap"] == "0.25") & (pairs["system"] == "aod+even+out")]["setting"].iloc[0]
    eps = []
    for comp in setting.split("+"):
        eps += R.episodes(build(comp, d, dtabs, it))
    eps = R.merge(eps)
    sched = M.schedule(it, 1)
    extra = sched.loc[~sched["regular"], "time"].to_numpy()
    starts = np.array([s for s, _ in eps], dtype="datetime64[ns]")
    ends = np.array([e for _, e in eps], dtype="datetime64[ns]")
    matched = near(starts, extra) | near(ends, extra)
    t_pat = d["arrival"].to_numpy()
    chance = float(near(t_pat, extra).mean())
    yr = pd.DatetimeIndex(starts).year
    for y in (2023, 2024, 2025, "all"):
        m = np.ones(starts.size, bool) if y == "all" else (yr == y)
        n = int(m.sum())
        conc.append(dict(item=it, year=y, episodes=n, matched=int(matched[m].sum()),
                         matched_share=float(matched[m].mean()) if n else np.nan, chance=chance,
                         enrichment=float(matched[m].mean() / chance) if n and chance else np.nan,
                         extra_qc=int(((pd.DatetimeIndex(extra).year == y) if y != "all" else np.ones(extra.size, bool)).sum())))
    # monthly alarms and QC median deviation from the 2023-24 median (%)
    mon = pd.Series(1, index=pd.DatetimeIndex(starts)).resample("MS").sum().reindex(
        pd.date_range("2023-01-01", "2025-12-01", freq="MS"), fill_value=0)
    qc = Q.load(it, 1)
    qc["lvl"] = qc["level"].astype(str).map({"L": "L", "H": "H", "1": "L", "2": "H"})
    dev = {lv: qc[(qc["lvl"] == lv) & (qc["time"] <= M.DEV_END)]["value"].median() for lv in ("L", "H")}
    for lv in ("L", "H"):
        g = qc[qc["lvl"] == lv].set_index("time")["value"].resample("MS").median()
        dev_pct = (g / dev[lv] - 1) * 100
        for mth, v in dev_pct.items():
            monthly.append(dict(item=it, month=mth, alarms=int(mon.get(mth, 0)), level=lv, qc_dev_pct=v))
    print(it, flush=True)

pd.DataFrame(conc).to_csv(ROOT / "outputs/tables/shadow_concordance.csv", index=False)
pd.DataFrame(monthly).to_csv(ROOT / "outputs/tables/shadow_monthly.csv", index=False)


# ---- 2 x 2 with item-level QC shifts (docs/16 layer 2) -----------------------------------------------
# QC shift event: on day t, for BOTH levels the median of days t..t+4 differs from the median of days
# t-10..t-1 by more than 2 x CVa (relative) in the SAME direction; consecutive days within 7 days = one event.
# PBRTQC event: 3-way alarm episode start. Concordant if the two occur within +/-3 days.
# Cells: both = analytical change; QC only = QC-material side (lot change, deterioration);
#        PBRTQC only = patient population / patient-only error.
SHIFT_K, MATCH = 2.0, np.timedelta64(3, "D")
cv = pd.read_csv(ROOT / "outputs/tables/iqc_totalcv.csv").set_index("item")["cva"]
rows2 = []
for it in spec.ITEMS:
    qc = Q.load(it, 1)
    qc["lvl"] = qc["level"].astype(str).map({"L": "L", "H": "H", "1": "L", "2": "H"})
    days = pd.date_range("2023-01-01", "2025-12-31")
    sc = {}
    for lv in ("L", "H"):
        dm = qc[qc["lvl"] == lv].groupby(qc["time"].dt.normalize())["value"].median().reindex(days)
        after = dm[::-1].rolling(5, min_periods=3).median()[::-1]
        before = dm.shift(1).rolling(10, min_periods=5).median()
        sc[lv] = (after / before - 1) / cv[it]
    ev_day = days[((sc["L"] > SHIFT_K) & (sc["H"] > SHIFT_K)) | ((sc["L"] < -SHIFT_K) & (sc["H"] < -SHIFT_K))]
    qev = []
    for dd in ev_day:
        if not qev or (dd - qev[-1]).days > 7:
            qev.append(dd)
    qev = np.array(qev, dtype="datetime64[ns]")
    d = lis.load(it, unit=1, start=R.DEV[0], end=R.VAL[1])
    d_all = lis.load(it, unit=None, start=R.DEV[0], end=R.VAL[1])
    dtabs = {G: R.delta_table(d_all, lookback=G) for G in R.AOD_LOOKBACK}
    setting = pairs[(pairs["item"] == it) & (pairs["cap"] == "0.25") & (pairs["system"] == "aod+even+out")]["setting"].iloc[0]
    eps = []
    for comp in setting.split("+"):
        eps += R.episodes(build(comp, d, dtabs, it))
    pst = np.array([s for s, _ in R.merge(eps)], dtype="datetime64[ns]")
    q_hit = near(qev, pst, MATCH) if qev.size and pst.size else np.zeros(qev.size, bool)
    p_hit = near(pst, qev, MATCH) if qev.size and pst.size else np.zeros(pst.size, bool)
    rows2.append(dict(item=it, qc_shifts=int(qev.size), both=int(q_hit.sum()), qc_only=int((~q_hit).sum()),
                      pb_episodes=int(pst.size), pb_only=int((~p_hit).sum()),
                      both_dates=";".join(str(x)[:10] for x in qev[q_hit])))
pd.DataFrame(rows2).to_csv(ROOT / "outputs/tables/shadow_2x2.csv", index=False)
