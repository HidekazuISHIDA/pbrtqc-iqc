"""Select PBRTQC settings on the development period, evaluate them on 2025 (docs/15 §3-§4).

Truncation "ref" = JCCLS common reference interval (sex-specific where defined; Mg 1.8-2.4 in-house).
Systems compared: in+out (MA on each stream), out+aod (MA on outpatients + average of patient deltas,
realsim.AoDStream, truncation 2.5-97.5 / none, lookback G 7/30/90 d), even+out (comparator: even check
R value with the same G and N candidates), aod+even+out (3-way OR under the same cap), select (per item,
AoD or even check, whichever is faster on the development period), pooled (single MA).

Development period (2023-2024), per item:
  for every stream (in / out / pooled) and setting (truncation x N): episodes (FAR) and, for 200
  onsets with a +/- 1 x TEa proportional bias, the hours from onset to the first alarm.
  The in+out system is chosen as a PAIR (in setting, out setting), because its alarm burden is the
  sum of both streams: among pairs whose merged FAR <= cap, take the one with the shortest median
  time to detection (ties: higher share within 24 h, then lower FAR). Pooled is chosen the same way.
  Caps (alarm episodes / week / item / analyzer): none, 1.0, 0.5, 0.25 (primary), 0.1. If no setting meets a cap, the
  lowest-FAR one is taken and flagged.
  The evaluation error is 1 x TEa until IQC data give SE* (HANDOVER A-3).
Validation (2025): chosen settings at 0.5 / 1 / 2 x TEa, per stream and for the OR system.

Outputs (aggregates only):
  outputs/tables/select_dev.csv   every stream x setting: development FAR and detection
  outputs/tables/select_pairs.csv chosen setting per item x cap (with development performance)
  outputs/tables/validate.csv     2025 performance per item x cap x k x stream
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
lis.VALUE = ARGS.get("value", "initial")
SUF = "" if lis.VALUE == "initial" else f"_{lis.VALUE}"
NOPRUNE = "--no-prune" in sys.argv     # check (2026-10-04): exhaustive three-component search without the per-component pre-filter
if NOPRUNE:
    SUF += "_noprune"
N_TRIALS = 200
KS = [0.5, 1.0, 2.0]
CAPS = {"none": np.inf, "1.0": 1.0, "0.5": 0.5, "0.25": 0.25, "0.1": 0.1}
S = spec.load()
rng = np.random.default_rng(20260929)
dev_rows, pair_rows, val_rows = [], [], []


def score(hours: np.ndarray) -> tuple:
    return (float(np.median(hours)), -float((hours <= 24).mean()))


for it in spec.ITEMS:
    tea = R.tea(it, S)
    d = lis.load(it, unit=1, start=R.DEV[0], end=R.VAL[1])
    masks = R.stream_masks(d)
    signs = np.where(np.arange(N_TRIALS) % 2 == 0, 1.0, -1.0)
    t_dev = R.onsets(d, R.DEV, N_TRIALS, rng)
    t_val = R.onsets(d, R.VAL, N_TRIALS, rng)
    t_all = d["arrival"].to_numpy()
    d_all = lis.load(it, unit=None, start=R.DEV[0], end=R.VAL[1])
    dtabs = {G: R.delta_table(d_all, lookback=G) for G in R.AOD_LOOKBACK}

    st, ep, hrs = {}, {}, {}
    for trunc in R.TRUNCS:
        for N in R.NS:
            for sname, m in masks.items():
                x = R.make_stream(d, m, it, trunc, N)
                if x is None:
                    continue
                key = (sname, trunc, N)
                st[key], ep[key] = x, R.episodes(x)
                nh = np.array([R.first_alarm_hours(x, t_dev[j], signs[j] * tea) for j in range(N_TRIALS)])
                hrs[key] = nh[:, 1]
                r = dict(item=it, stream=sname, trunc=trunc, N=N, far_dev_wk=R.rate_per_week(ep[key], R.DEV))
                r.update({f"{k}_dev": v for k, v in R.summarise(nh).items()})
                dev_rows.append(r)
    for G in R.AOD_LOOKBACK:
      for trunc in R.AOD_TRUNCS:
        for N in R.NS:
            key = ("aod", f"{trunc}|G{G}", N)
            x = R.make_aod_stream(dtabs[G], trunc, N)
            st[key], ep[key] = x, R.episodes(x)
            nh = np.array([R.first_alarm_hours(x, t_dev[j], signs[j] * tea) for j in range(N_TRIALS)])
            hrs[key] = nh[:, 1]
            r = dict(item=it, stream="aod", trunc=f"{trunc}|G{G}", N=N, far_dev_wk=R.rate_per_week(ep[key], R.DEV))
            r.update({f"{k}_dev": v for k, v in R.summarise(nh).items()})
            dev_rows.append(r)
    for G in R.AOD_LOOKBACK:             # comparator: even check, same lookback and N candidates
        for N in R.NS:
            key = ("even", f"G{G}", N)
            x = R.make_aod_stream(dtabs[G], "none", N, kind="even")
            st[key], ep[key] = x, R.episodes(x)
            nh = np.array([R.first_alarm_hours(x, t_dev[j], signs[j] * tea) for j in range(N_TRIALS)])
            hrs[key] = nh[:, 1]
            r = dict(item=it, stream="even", trunc=f"G{G}", N=N, far_dev_wk=R.rate_per_week(ep[key], R.DEV))
            r.update({f"{k}_dev": v for k, v in R.summarise(nh).items()})
            dev_rows.append(r)

    # candidate systems: tuples of streams alarming with OR; alarm burden = merged episodes
    by = {n: [k for k in st if k[0] == n] for n in ("in", "out", "aod", "even", "pooled")}

    def combos(*names):
        import itertools
        out = []
        for keys in itertools.product(*(by[n] for n in names)):
            far = R.rate_per_week(R.merge(sum((ep[k] for k in keys), [])), R.DEV)
            h = hrs[keys[0]]
            for k in keys[1:]:
                h = np.minimum(h, hrs[k])
            out.append((keys, far, h))
        return out

    cand = {"in+out": combos("in", "out"), "aod+out": combos("aod", "out"), "even+out": combos("even", "out"),
            "pooled": combos("pooled")}
    # 3-way combination (AoD + even check + outpatient MA): prune components that alone exceed the cap
    three = {}
    full3 = combos("aod", "even", "out") if NOPRUNE else None
    for cap_name, cap in CAPS.items():
        if NOPRUNE:                     # merging episodes < 8 h apart can lower the count of a union, so test without the filter
            three[cap_name] = full3
            continue
        keep = {n: [k for k in by[n] if R.rate_per_week(ep[k], R.DEV) <= cap] or by[n] for n in ("aod", "even", "out")}
        saved = {n: by[n] for n in keep}
        by.update(keep)
        three[cap_name] = combos("aod", "even", "out")
        by.update(saved)

    for cap_name, cap in CAPS.items():
        chosen = {}
        cands_all = dict(cand, **{"aod+even+out": three[cap_name]})
        for system, cands in cands_all.items():
            ok = [c for c in cands if c[1] <= cap]
            met = bool(ok)
            if not ok:
                ok = [min(cands, key=lambda c: c[1])]
            best = min(ok, key=lambda c: (*score(c[2]), c[1]))
            chosen[system] = best
            keys, far, h = best
            pair_rows.append(dict(item=it, cap=cap_name, system=system, cap_met=met,
                                  setting="+".join(f"{x[0]}:{x[1]}/N{x[2]}" for x in keys), far_dev_wk=far,
                                  med_hours_dev=float(np.median(h)), p_det_24h_dev=float((h <= 24).mean())))
        # per-item choice between AoD and even check, decided on the development period
        pick = min(("aod+out", "even+out"), key=lambda s: (*score(chosen[s][2]), chosen[s][1]))
        chosen["select"] = chosen[pick]

        # ---- validation (2025)
        systems = {s: list(chosen[s][0]) for s in chosen}
        for s in ("in+out", "aod+out", "even+out"):
            for x in chosen[s][0]:
                systems.setdefault(f"{x[0]}({s})", [x])
        keys_needed = {x for v in systems.values() for x in v}
        for k in KS:
            res = {s: [] for s in systems}
            for j in range(N_TRIALS):
                hh = {key: R.first_alarm_hours(st[key], t_val[j], signs[j] * k * tea) for key in keys_needed}
                for s, keys in systems.items():
                    if len(keys) == 1:
                        res[s].append(hh[keys[0]])
                        continue
                    h = min(hh[x][1] for x in keys)
                    if np.isfinite(h):
                        t_det = t_val[j] + np.timedelta64(int(h * 3600), "s")
                        n = float(np.searchsorted(t_all, t_det, "right") - np.searchsorted(t_all, t_val[j]))
                    else:
                        n = np.inf
                    res[s].append((n, h))
            for s, keys in systems.items():
                e = R.merge(sum((ep[x] for x in keys), []))
                r = dict(item=it, cap=cap_name, stream=s, k=k, tea=tea,
                         setting="+".join(f"{x[0]}:{x[1]}/N{x[2]}" for x in keys),
                         far_dev_wk=R.rate_per_week(e, R.DEV), rate_val_wk=R.rate_per_week(e, R.VAL))
                r.update(R.summarise(np.array(res[s], float)))
                val_rows.append(r)
    print(it, flush=True)

pd.DataFrame(dev_rows).replace(np.inf, np.nan).to_csv(ROOT / f"outputs/tables/select_dev{SUF}.csv", index=False)
pd.DataFrame(pair_rows).replace(np.inf, np.nan).to_csv(ROOT / f"outputs/tables/select_pairs{SUF}.csv", index=False)
pd.DataFrame(val_rows).replace(np.inf, np.nan).to_csv(ROOT / f"outputs/tables/validate{SUF}.csv", index=False)
