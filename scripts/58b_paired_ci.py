"""Paired bootstrap intervals for MaxE(Nuf) comparisons in the five low-sigma items (common random numbers).

Within one run of scripts/53 all systems share error onsets, signs and QC draws, so trials are resampled jointly
(same indices for every system and error size). MaxE(Nuf) = max over k of the mean per-trial contribution.
Inputs: fusion_trials.npz, fusion_loo_trials.npz. Output: outputs/tables/paired_ci.csv
(ratio = MaxE(Nuf) of system A / system B, 95 % percentile interval, 2000 resamples).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import spec

T = ROOT / "outputs/tables"
LOW5 = ["Na", "ALB", "Cl", "Ca", "Mg"]
B = 2000
rng = np.random.default_rng(1)
COMPARE = {"fusion": [("aod+even+out", "IQC"), ("aod+even+out", "aod+out"), ("aod+even+out", "even+out"),
                      ("aod+even+out", "in+out"), ("aod+even+out", "pooled")],
           "fusion_loo": [("no_aod", "full"), ("no_even", "full"), ("no_out", "full")]}
rows = []
for run, pairs in COMPARE.items():
    z = np.load(T / f"{run}_trials.npz")
    for it in LOW5:
        ks = sorted({float(k.split("|")[2]) for k in z.files if k.startswith(it + "|")})
        sysn = {s for a_, b_ in pairs for s in (a_, b_)}
        c = {s: np.stack([z[f"{it}|{s}|{k:g}" if f"{it}|{s}|{k:g}" in z.files else f"{it}|{s}|{k}"][0] for k in ks]) for s in sysn}
        n = next(iter(c.values())).shape[1]
        idx = rng.integers(0, n, (B, n))
        boot = {s: np.stack([v[:, i].mean(1) for i in idx]).max(1) for s, v in c.items()}   # (B,)
        point = {s: v.mean(1).max() for s, v in c.items()}
        for a_, b_ in pairs:
            r = boot[a_] / boot[b_]
            rows.append(dict(run=run, item=spec.disp(it), A=a_, B=b_, maxenuf_A=point[a_], maxenuf_B=point[b_],
                             ratio=point[a_] / point[b_], lo=np.percentile(r, 2.5), hi=np.percentile(r, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(T / "paired_ci.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
