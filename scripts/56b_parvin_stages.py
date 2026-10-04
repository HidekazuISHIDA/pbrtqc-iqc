"""Staged check of the E(Nuf) accounting against Parvin's formula (synthetic; no patient data).

Parvin CA. Clin Chem 2008;54:2049-54 (continuous mode, bracketed QC, persistent error):
    E(Nuf) = dPE [ q E(N0) + (q^2 / Ped) N ],  q = 1 - Ped
which assumes a constant probability Ped of rejection at every QC event after the shift. The study simulation
(scripts/53) differs in three ways, added here one at a time with the same counting rule as scripts/53
(results between onset and the last accepted QC event before rejection are final):

  A  constant run size N, memoryless rule (1-3s on either of two levels), no follow-up limit  -> should equal the formula
  B  as A, standard multirule 1-3s/2-2s/R-4s/4-1s/10x with 10 in-control events of history (across-event memory)
  C  as B, follow-up limited to 34 QC events (14 days at the median 17 events per week)
  D  real QC times and patient arrivals, 2025 (outputs/tables/parvin_check.csv, from scripts/56)

The formula in B and C uses Ped at the first event after the shift (iqc.power), as in scripts/56.
Counts are per unit dPE (dPE cancels in the ratio). Output: outputs/tables/parvin_stages.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import iqc

rng = np.random.default_rng(20261004)
N, TRIALS, HIST, FOLLOW, CAP = 30, 4000, 10, 34, 5000
SHIFTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def formula(ped):
    q = 1 - ped
    return q * N / 2 + q * q / ped * N


def finals(shift, rules, follow):
    out = np.empty(TRIALS)
    for j in range(TRIALS):
        r0 = N * rng.random()                      # results between onset and the first QC event
        z = list(rng.normal(0, 1, (HIST, 2)))
        m = 0                                      # accepted events after onset
        while m < follow:
            z.append(rng.normal(shift, 1, 2))
            if iqc._violates(np.array(z[-12:]), rules):
                break
            m += 1
        out[j] = 0.0 if m == 0 else r0 + (m - 1) * N
    return out.mean()


rows = []
for s in SHIFTS:
    p_level = norm.sf(3 - s) + norm.cdf(-3 - s)    # 1-3s on one level
    p1 = 1 - (1 - p_level) ** 2                     # either of two levels
    ped_m = iqc.power(s, n_sim=20000, hist=HIST)
    a = finals(s, ("1-3s",), CAP)
    b = finals(s, iqc.RULE_SET_DEFAULT, CAP)
    c = finals(s, iqc.RULE_SET_DEFAULT, FOLLOW)
    rows.append(dict(shift_sd=s, ped_13s=p1, ped_multirule_first=ped_m,
                     A_sim=a, A_formula=formula(p1), A_ratio=a / formula(p1),
                     B_sim=b, B_formula=formula(ped_m), B_ratio=b / formula(ped_m),
                     C_sim=c, C_ratio=c / formula(ped_m)))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
pv = pd.read_csv(ROOT / "outputs/tables/parvin_check.csv")
pv["ratio"] = pv["e_nuf_sim"] / pv["e_nuf_formula"]
d = pv.groupby("k")["ratio"].median()
out.attrs["D"] = d.to_dict()
out.to_csv(ROOT / "outputs/tables/parvin_stages.csv", index=False)
print("D (real schedule, median over items) by k:", d.round(2).to_dict())
