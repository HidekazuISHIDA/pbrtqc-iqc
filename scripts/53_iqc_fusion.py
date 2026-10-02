"""IQC + PBRTQC fusion on the real 2025 data: detection and patient risk E(Nuf) (docs/16).

Per item (analyzer #1) and error size k x TEa (proportional, +/- alternating), 300 onsets drawn uniformly over
2025 patient results (14-day follow-up):

  IQC (S1)   the REAL QC event times of the unit (regular + extra); QC values simulated as z ~ N(0,1) per
             level plus the shift k*TEa / CV_level after onset (CV = robust quarterly Total CV,
             iqc_model.total_cv); standard Westgard multirule 1-3s/2-2s/R-4s/4-1s/10x with 10 in-control
             events of history. Detection = first rejecting event.
  PBRTQC     the settings chosen on 2023-24 under the 0.25 alarms/week cap (outputs/tables/select_pairs.csv),
             same error injected into the patient stream (realsim).
  S3         IQC OR PBRTQC: detection at the earlier of the two.

Final erroneous results: on detection the laboratory recalls results back to the last ACCEPTED QC event
after onset (docs/16 assumption; Parvin 2008 bracketed-QC accounting), so results released before that
event stay final. E(Nuf) = dPE(k) x mean number of final results after onset, dPE from risk.delta_pe with
sigma = TEa / CVa (bias = 0 until EQA data arrive, HANDOVER A-1). Undetected at 14 days = counted up to the
last accepted QC (lower bound).

Also: in-control IQC false-rejection rate on the 2023-24 schedule, and an analytic check of the S1 E(Nuf)
against Parvin's formula with the empirical run sizes.

Sensitivity (--eqa-bias): a constant laboratory bias b0 from the JAMT 2026 survey (mean of 2 samples,
data/EQA/parsed/jamt2026.csv) is added: dPE = P(|b0 + shift + e| > TEa) - P(|b0 + e| > TEa), computed per sign
of the injected shift. Output then goes to outputs/tables/fusion_eqa.csv.

What-if runs for the Discussion (no change to the actual operation is implied):
  --cap=X          PBRTQC settings chosen under a different alarm cap (select_pairs.csv must contain it)
  --extra-qc=17,1  add one QC event per day at each of these clock hours to the real schedule
  --items=Na,ALB   restrict to these items
  Output: outputs/tables/fusion_whatif_cap<X>_qc<hours>.csv (IQC and IQC + aod+even+out only).

Outputs (aggregates only): outputs/tables/iqc_totalcv.csv, outputs/tables/fusion.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R, iqc_model as M, iqc, risk

R.FOLLOW = np.timedelta64(14, "D")
EQA = "--eqa-bias" in sys.argv
B0 = (pd.read_csv(ROOT / "data/EQA/parsed/jamt2026.csv").groupby("item")["bias_pct"].mean() / 100).to_dict() \
    if EQA else {}
N_TRIALS = 300
KS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
UNIT = int(ARGS.get("unit", 1))                    # sensitivity: apply unit-1 settings to unit 2
lis.VALUE = ARGS.get("value", "initial")           # sensitivity: final reported values
CAP = ARGS.get("cap", "0.25")
EXTRA_QC = [int(h) for h in ARGS["extra-qc"].split(",")] if "extra-qc" in ARGS else []
ITEMS = ARGS["items"].split(",") if "items" in ARGS else spec.ITEMS
WHATIF = bool({"cap", "extra-qc", "items"} & set(ARGS))
SENS = UNIT != 1 or lis.VALUE != "initial"
SYSTEMS = ["aod+even+out"] if WHATIF else ["aod+even+out", "aod+out", "even+out", "in+out", "pooled"]
HIST = 10
S = spec.load()
rng = np.random.default_rng(20260930)
pairs = pd.read_csv(ROOT / f"outputs/tables/select_pairs{'' if ARGS.get('value', 'initial') == 'initial' else '_' + ARGS['value']}.csv")
cv_rows, rows = [], []


def build(key, d, dtabs, item):
    kind, rest = key.split(":")
    trunc, N = rest.rsplit("/N", 1)
    N = int(N)
    if kind in ("in", "out", "pooled"):
        return R.make_stream(d, R.stream_masks(d)[kind], item, trunc, N)
    if kind == "aod":
        tr, G = trunc.split("|G")
        return R.make_aod_stream(dtabs[int(G)], tr, N, unit=UNIT)
    return R.make_aod_stream(dtabs[int(trunc[1:])], "none", N, unit=UNIT, kind="even")


def iqc_detect(ev_t, t_on, zL, zH, sL, sH):
    """Time of the first Westgard rejection after onset, or None. zL/zH: HIST + len(ev_t) in-control draws."""
    z = np.column_stack([zL, zH])
    z[HIST:, 0] += sL
    z[HIST:, 1] += sH
    for i in range(len(ev_t)):
        if iqc._violates(z[: HIST + i + 1], iqc.RULE_SET_DEFAULT):
            return ev_t[i]
    return None


for it in ITEMS:
    tea = R.tea(it, S)
    cv = M.total_cv(it, UNIT)
    cva = cv["mean"]
    sched = M.schedule(it, UNIT)
    ev_all = sched["time"].to_numpy()
    if EXTRA_QC:
        days = pd.date_range("2023-01-01", "2025-12-31")
        extra = np.concatenate([(days + pd.Timedelta(hours=h)).to_numpy() for h in EXTRA_QC])
        ev_all = np.sort(np.concatenate([ev_all, extra]))
    # in-control IQC false rejections on the development schedule
    dev_ev = ev_all[ev_all <= np.datetime64(M.DEV_END)]
    zc = rng.normal(0, 1, (dev_ev.size, 2))
    far_iqc = iqc.evaluate_series(zc).sum() / R.weeks(*R.DEV) if hasattr(R, "weeks") else \
        iqc.evaluate_series(zc).sum() / (((pd.Timestamp(R.DEV[1]) - pd.Timestamp(R.DEV[0])).days + 1) / 7)
    cv_rows.append(dict(item=it, cv_L=cv["L"], cv_H=cv["H"], cva=cva, tea=tea, sigma=tea / cva,
                        qc_events_per_week=dev_ev.size / 104.4, iqc_false_rej_wk=far_iqc))

    d = lis.load(it, unit=UNIT, start=R.DEV[0], end=R.VAL[1])
    d_all = lis.load(it, unit=None, start=R.DEV[0], end=R.VAL[1])
    dtabs = {G: R.delta_table(d_all, unit=UNIT, lookback=G) for G in R.AOD_LOOKBACK}
    t_pat = d["arrival"].to_numpy()
    ok = ((d["date"] >= R.VAL[0]) & (d["date"] <= pd.Timestamp(R.VAL[1]) - pd.Timedelta(days=14))).to_numpy()
    t_on = t_pat[rng.choice(np.flatnonzero(ok), N_TRIALS)]
    signs = np.where(np.arange(N_TRIALS) % 2 == 0, 1.0, -1.0)

    sel = pairs[(pairs["item"] == it) & (pairs["cap"] == CAP)].set_index("system")["setting"]
    comps = {s: sel[s].split("+") for s in SYSTEMS}
    streams = {k: build(k, d, dtabs, it) for k in {c for v in comps.values() for c in v}}
    eps = {k: R.episodes(st) for k, st in streams.items()}
    far = {}
    for s_ in SYSTEMS:
        e_ = R.merge(sum((eps[c] for c in comps[s_]), []))
        far[s_] = (R.rate_per_week(e_, R.DEV), R.rate_per_week(e_, R.VAL))

    # QC events in each trial's window + in-control history draws (shared by all systems and k)
    win = [(np.searchsorted(ev_all, t, "right"), np.searchsorted(ev_all, t + R.FOLLOW, "right")) for t in t_on]
    zdraw = [rng.normal(0, 1, (HIST + b - a, 2)) for a, b in win]

    for k in KS:
        bias = k * tea
        b0 = B0.get(it, 0.0)
        dpe_s = {sg: max(0.0, risk.p_exceed(tea / cva, (b0 + sg * bias) / cva) - risk.p_exceed(tea / cva, b0 / cva))
                 for sg in (1.0, -1.0)}
        dpe_t = np.array([dpe_s[sg] for sg in signs])
        dpe = float(dpe_t.mean())
        det = {s: np.full(N_TRIALS, np.inf) for s in ["IQC"] + SYSTEMS + [f"PB:{s}" for s in SYSTEMS]}
        finals = {s: np.zeros(N_TRIALS) for s in det}
        for j in range(N_TRIALS):
            b = signs[j] * bias
            a0, a1 = win[j]
            ev_t = ev_all[a0:a1]
            td_iqc = iqc_detect(ev_t, t_on[j], zdraw[j][:, 0].copy(), zdraw[j][:, 1].copy(), b / cv["L"], b / cv["H"])
            hc = {c: R.first_alarm_hours(streams[c], t_on[j], b)[1] for c in streams}
            tds = {"IQC": td_iqc}
            for s in SYSTEMS:
                h = min(hc[c] for c in comps[s])
                t_pb = t_on[j] + np.timedelta64(int(h * 3600), "s") if np.isfinite(h) else None
                tds[f"PB:{s}"] = t_pb
                cand = [x for x in (td_iqc, t_pb) if x is not None]
                tds[s] = min(cand) if cand else None
            for s, td in tds.items():
                end = td if td is not None else t_on[j] + R.FOLLOW
                if td is not None:
                    det[s][j] = (td - t_on[j]) / np.timedelta64(1, "h")
                # last accepted QC event after onset and before detection (IQC-detecting event excluded)
                acc = ev_t[(ev_t > t_on[j]) & (ev_t < end)]
                if acc.size:
                    finals[s][j] = np.searchsorted(t_pat, acc[-1], "left") - np.searchsorted(t_pat, t_on[j], "left")
        for s in det:
            h = det[s]
            base = s[3:] if s.startswith("PB:") else s
            fd, fv = far.get(base, (np.nan, np.nan)) if s != "IQC" else (np.nan, np.nan)
            rows.append(dict(item=it, system=s, k=k, unit=UNIT, value=lis.VALUE, far_dev_wk=fd, rate_val_wk=fv,
                             dpe=dpe, b0=b0, mean_finals=float(finals[s].mean()),
                             e_nuf=float((dpe_t * finals[s]).mean()), det_rate=float(np.isfinite(h).mean()),
                             med_hours=float(np.median(h)), p_det_24h=float((h <= 24).mean())))
    print(it, flush=True)

name = "fusion_eqa" if EQA else "fusion"
if SENS:
    name = f"fusion_sens_unit{UNIT}_{lis.VALUE}"
if WHATIF:
    name = f"fusion_whatif_cap{CAP}_qc{'-'.join(map(str, EXTRA_QC)) or 'none'}"
    pd.DataFrame(cv_rows).to_csv(ROOT / f"outputs/tables/{name}_iqc.csv", index=False)
elif SENS:
    pd.DataFrame(cv_rows).to_csv(ROOT / f"outputs/tables/{name}_iqc.csv", index=False)
elif not EQA:
    pd.DataFrame(cv_rows).to_csv(ROOT / "outputs/tables/iqc_totalcv.csv", index=False)
pd.DataFrame(rows).replace(np.inf, np.nan).to_csv(ROOT / f"outputs/tables/{name}.csv", index=False)
