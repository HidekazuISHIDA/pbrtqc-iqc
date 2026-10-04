"""Bias sensitivity analysis split by the sign of the injected error (five low-sigma items).

In scripts/53, Delta P_E = max(0, P(|e + b + s| > TEa) - P(|e + b| > TEa)) for each sign of the shift s; with a constant bias b,
a shift opposite to b lowers the exceedance probability and is counted as 0 (no credit for improvement). Onsets alternate in
sign (even trial index = positive), so the per-trial contributions in <run>_trials.npz can be split by sign.
Output: outputs/tables/bias_by_sign.csv (MaxE(Nuf) per sign; positive = same direction as the bias).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import spec

T = ROOT / "outputs/tables"
rows = []
for run, lab in [("fusion", "No bias"), ("fusion_sens_unit1_initial_bias0.25", "Bias 0.25 × TEa")]:
    z = np.load(T / f"{run}_trials.npz")
    for it in ["Na", "ALB", "Cl", "Ca", "Mg"]:
        for s in ["IQC", "aod+even+out"]:
            keys = [k for k in z.files if k.startswith(f"{it}|{s}|")]
            c = np.stack([z[k][0] for k in keys])                     # (k, trials)
            pos, neg = c[:, 0::2].mean(1).max(), c[:, 1::2].mean(1).max()
            rows.append(dict(analysis=lab, item=spec.disp(it), system=s, maxenuf_pos=pos, maxenuf_neg=neg, maxenuf_pooled=c.mean(1).max()))
out = pd.DataFrame(rows)
out.to_csv(T / "bias_by_sign.csv", index=False)
print(out.round(1).to_string(index=False))
