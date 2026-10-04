"""Staged check of the E(Nuf) accounting against Parvin's formula (synthetic; no patient data).

Parvin CA. Clin Chem 2008;54:2049-54 (continuous mode, bracketed QC, persistent error):
    E(Nuf) = dPE [ q E(N0) + (q^2 / Ped) N ],  q = 1 - Ped
which assumes a constant probability Ped of rejection at every QC event after the shift. The study simulation
(scripts/53) differs in three ways, added here one at a time with the same counting rule as scripts/53
(results between onset and the last accepted QC event before rejection are final):

  A  constant run size N, memoryless rule (1-3s on either of two levels), no follow-up limit  -> should equal the formula
  B  as A, standard multirule 1-3s/2-2s/R-4s/4-1s/10x with 10 in-control events of history (across-event memory)
  C  as B on the same trajectories, follow-up limited to 34 QC events (14 days at the median 17 events per week)
  D  real QC times and patient arrivals, 2025 (outputs/tables/parvin_check.csv, from scripts/56)

Formula: A uses the exact 1-3s rejection probability; B and C use Ped at the first event after the shift (iqc.power, multirule
with 10 in-control events of history), as in scripts/56. 20 000 trials per shift; standard errors are reported.
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
N, TRIALS, HIST, FOLLOW, CAP = 30, 20000, 10, 34, 5000
SHIFTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def formula(ped):
    q = 1 - ped
    return q * N / 2 + q * q / ped * N


def accepted_events(shift, rules):
    """For each trial: r0 (results before the first QC event) and m (accepted events after onset before rejection,
    capped at CAP). One trajectory per trial, so B (unlimited) and C (14-day follow-up) are computed on the same draws."""
    r0 = N * rng.random(TRIALS)
    m = np.empty(TRIALS, int)
    for j in range(TRIALS):
        z = list(rng.normal(0, 1, (HIST, 2)))
        k = 0
        while k < CAP:
            z.append(rng.normal(shift, 1, 2))
            if iqc._violates(np.array(z[-12:]), rules):
                break
            k += 1
        m[j] = k
    return r0, m


def finals(r0, m, follow=None):
    """Results between onset and the last accepted QC event before rejection (scripts/53 counting rule); with a follow-up
    limit, an error still undetected after `follow` events counts results up to the last accepted event within follow-up."""
    mm = m if follow is None else np.minimum(m, follow)
    return np.where(mm == 0, 0.0, r0 + (mm - 1) * N)


def mean_se(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(x.size)


rows = []
for s in SHIFTS:
    p_level = norm.sf(3 - s) + norm.cdf(-3 - s)    # 1-3s on one level
    p1 = 1 - (1 - p_level) ** 2                     # either of two levels (exact)
    ped_m = iqc.power(s, n_sim=40000, hist=HIST)    # multirule, first event after the shift (simulated)
    ra, ma = accepted_events(s, ("1-3s",))
    rb, mb = accepted_events(s, iqc.RULE_SET_DEFAULT)
    a, a_se = mean_se(finals(ra, ma))
    b, b_se = mean_se(finals(rb, mb))
    c, c_se = mean_se(finals(rb, mb, FOLLOW))
    fa, fb = formula(p1), formula(ped_m)
    rows.append(dict(shift_sd=s, ped_13s=p1, ped_multirule_first=ped_m, trials=TRIALS,
                     A_formula=fa, A_sim=a, A_se=a_se, A_ratio=a / fa, A_ratio_lo=(a - 1.96 * a_se) / fa, A_ratio_hi=(a + 1.96 * a_se) / fa,
                     B_formula=fb, B_sim=b, B_se=b_se, B_ratio=b / fb, C_sim=c, C_se=c_se, C_ratio=c / fb,
                     share_censored_C=float((mb >= FOLLOW).mean())))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs/tables/parvin_stages.csv", index=False)
pv = pd.read_csv(ROOT / "outputs/tables/parvin_check.csv")
pv = pv[pv["e_nuf_formula"] > 0.01]
print("D (real schedule; items with formula E(Nuf) > 0.01):",
      {k: (round(g["e_nuf_sim"].div(g["e_nuf_formula"]).median(), 2), len(g)) for k, g in pv.groupby("k")})
