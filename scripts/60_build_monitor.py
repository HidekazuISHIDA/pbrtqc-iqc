"""Build the data for the local monitoring screen (outputs/monitor/, docs/18).

For every item (analyzer #1) and every hour of 2025, the score of each component
    s = (statistic - centre) / (3 SD)        (+-1 = control limit, sign kept)
Components: QC L, QC H (real QC values), AoD, even check, outpatient MA (settings chosen on 2023-24
under the 0.25/week cap, select_pairs.csv). The value shown for an hour is the last value available
at the end of that hour (carried forward).

QC centre: median of the regular QC of the previous 30 days (display only; see qc_scores);
SD = robust Total CV x centre.
GLU uses Multiqual (its QAP series stops in 2025-09).

Aggregates only (moving statistics over >= 20 results, QC values) - no patient-level values leave the
statistics; the page is served on 127.0.0.1 only.
Output: outputs/monitor/data.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R, iqc_model as M, iqc_data as Q

HOURS = pd.date_range("2025-01-01 00:00", "2025-12-31 23:00", freq="h")
S = spec.load()
pairs = pd.read_csv(ROOT / "outputs/tables/select_pairs.csv")
cvtab = pd.read_csv(ROOT / "outputs/tables/iqc_totalcv.csv").set_index("item")
COMP = ["qcL", "qcH", "aod", "even", "out"]


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


def hourly(t, v):
    """Last value at or before the end of each hour of 2025 (NaN before the first)."""
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    ends = (HOURS + pd.Timedelta(hours=1)).to_numpy()
    i = np.searchsorted(t, ends, "right") - 1
    out = np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)
    return out


def rnd(a):
    return [None if not np.isfinite(x) else round(float(x), 2) for x in a]


def qc_scores(item):
    """QC centre for the display = median of the REGULAR QC of the previous 30 days (up to the day before).
    Fixed lot centres are not usable here: lot numbers are missing and the QC drifts within a lot, so fixed
    centres leave e.g. K (H), TG, T-CHO outside the limits for most of 2025. A sudden change stays visible
    for about two weeks; persistent errors are the PBRTQC components' job."""
    material = "Multiqual" if item == "GLU" else "QAP"
    d = Q.load(item, 1, material=material)
    sched = M.schedule(item, 1)
    reg_times = set(sched.loc[sched["regular"], "time"].dt.floor("h"))
    d["regular"] = d["time"].dt.floor("h").isin(reg_times)
    days = pd.date_range("2022-12-01", "2025-12-31")
    out = {}
    for lv, key in zip(sorted(d["level"].astype(str).unique()), ("qcH", "qcL") if material == "QAP" else ("qcL", "qcH")):
        x = d[d["level"].astype(str) == lv].sort_values("time").copy()
        lvl = "L" if key == "qcL" else "H"
        daily = x[x["regular"]].groupby(x["time"].dt.normalize())["value"].median().reindex(days)
        cen = daily.shift(1).rolling(30, min_periods=10).median()
        c = x["time"].dt.normalize().map(cen).to_numpy(float)
        sd = cvtab.loc[item, f"cv_{lvl}"] * c
        z = (x["value"].to_numpy(float) - c) / sd
        out[key] = dict(score=hourly(x["time"].to_numpy(), z / 3), centre_now=float(c[-1]), sd_now=float(sd[-1]))
    return out, []


data = dict(hours=[HOURS[0].isoformat(), len(HOURS)], comps=COMP, items=[])
for it in spec.ITEMS:
    d = lis.load(it, unit=1, start=R.DEV[0], end=R.VAL[1])
    d_all = lis.load(it, unit=None, start=R.DEV[0], end=R.VAL[1])
    dtabs = {G: R.delta_table(d_all, lookback=G) for G in R.AOD_LOOKBACK}
    setting = pairs[(pairs["item"] == it) & (pairs["cap"] == "0.25") & (pairs["system"] == "aod+even+out")]["setting"].iloc[0]
    meta = dict(item=spec.disp(it), key=it, unit=S[it]["unit"], analyzer=S[it]["analyzer"] + " #1", tea=round(R.tea(it, S) * 100, 2),
                cva=round(cvtab.loc[it, "cva"] * 100, 2), sigma=round(cvtab.loc[it, "sigma"], 1), setting=setting,
                comps={})
    scores, episodes = {}, []
    for comp in setting.split("+"):
        st = build(comp, d, dtabs, it)
        kind = comp.split(":")[0]
        mid, half = (st.cl[0] + st.cl[1]) / 2, (st.cl[1] - st.cl[0]) / 2
        scores[kind] = hourly(st.t, (st.stat - mid) / half)
        meta["comps"][kind] = dict(setting=comp.split(":", 1)[1], centre=float(mid), half=float(half))
        for s_, e_ in R.episodes(st):
            if s_ >= np.datetime64("2025-01-01"):
                episodes.append(dict(c=kind, s=str(s_)[:16], e=str(e_)[:16]))
    qc, cuts = qc_scores(it)
    for k, v in qc.items():
        scores[k] = v["score"]
        meta["comps"][k] = dict(setting="QAP" if it != "GLU" else "Multiqual", centre=v["centre_now"], half=3 * v["sd_now"])
    meta["lot_changes"] = cuts
    meta["scores"] = {k: rnd(scores[k]) for k in COMP}
    meta["episodes"] = sorted(episodes, key=lambda x: x["s"])
    data["items"].append(meta)
    print(it, len(episodes), flush=True)

out = ROOT / "outputs/monitor"
out.mkdir(parents=True, exist_ok=True)
(out / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
print("written", (out / "data.json").stat().st_size // 1024, "KB")
