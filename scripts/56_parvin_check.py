"""Check of the simulated IQC-only E(Nuf) against Parvin's formula (Figure S5).

Parvin CA. Clin Chem 2008;54:2049-54: continuous mode, bracketed QC, persistent error
    E(NU) = dPE {(ARL - 1) E(N_B) - (1 - P1)[E(N_B) - E(N_0)]},  ARL = 1/Ped, P1 = Ped
with E(N_B) = mean number of patient results between QC events in 2025 (analyzer #1) and
E(N_0) = N_eff / 2, N_eff = E[N^2]/E[N] (risk.e_nuf with n_onset). Ped = rejection probability of the
standard Westgard multirule at the first QC event after a shift (iqc.power, independent in-control history).
Simulation values come from outputs/tables/fusion.csv (system "IQC").
Output: outputs/tables/parvin_check.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R, iqc_model as M, iqc, risk

S = spec.load()
f = pd.read_csv(ROOT / "outputs/tables/fusion.csv")
f = f[f["system"] == "IQC"]
rows = []
for it in spec.ITEMS:
    tea = R.tea(it, S)
    cv = M.total_cv(it, 1)
    ev = M.schedule(it, 1)["time"].to_numpy()
    ev = ev[ev >= np.datetime64("2025-01-01")]
    t = lis.load(it, unit=1, start=R.VAL[0], end=R.VAL[1])["arrival"].to_numpy()
    n = np.diff(np.searchsorted(t, ev))
    nbar, neff = n.mean(), (n ** 2).sum() / n.sum()
    for k in sorted(f["k"].unique()):
        b = k * tea
        ped = iqc.power((b / cv["L"] + b / cv["H"]) / 2, n_sim=4000)
        formula = risk.e_nuf(tea / cv["mean"], b / cv["mean"], nbar, ped, n_onset=neff)
        sim = f[(f["item"] == it) & (f["k"] == k)]["e_nuf"].iloc[0]
        rows.append(dict(item=it, k=k, ped=ped, run_mean=nbar, run_neff=neff, e_nuf_formula=formula, e_nuf_sim=sim))
    print(it, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/parvin_check.csv", index=False)
