"""Contribution of each component: summary of outputs/tables/fusion_loo.csv (scripts/53 --loo).

Per item: median time to detection at 1 x TEa and MaxE(Nuf) over k for
IQC alone, the full system (IQC + AoD + even check + outpatient MA), the full system without one component
(settings unchanged), and PBRTQC without IQC. Check: the full system reproduces fusion.csv (aod+even+out).
Output: outputs/tables/loo_summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import spec

T = ROOT / "outputs/tables"
f = pd.read_csv(T / "fusion_loo.csv")
SYS = {"IQC": "IQC", "full": "Full", "no_aod": "−AoD", "no_even": "−even", "no_out": "−outMA", "PB:full": "−IQC"}
k1 = f[np.isclose(f["k"], 1.0)].pivot(index="item", columns="system", values="med_hours")
mx = f.groupby(["item", "system"])["e_nuf"].max().unstack()
order = [i for i in spec.ITEMS if i in k1.index]
out = pd.concat({"det_h": k1.loc[order, list(SYS)], "maxenuf": mx.loc[order, list(SYS)]}, axis=1)
out.columns = [f"{a}_{SYS[b]}" for a, b in out.columns]
out.index = [spec.disp(i) for i in order]
out.to_csv(T / "loo_summary.csv")

# reproducibility check against the main run
m = pd.read_csv(T / "fusion.csv")
m = m[m["system"] == "aod+even+out"].groupby("item")["e_nuf"].max()
d = (mx["full"] - m.reindex(mx.index)).abs().max()
print(f"max |MaxE(Nuf) full - main run| = {d:.4g}")
pd.set_option("display.width", 200)
print(out.round(2).to_string())
low5 = ["Na", "Alb", "Cl", "Ca", "Mg"]
print("\nmedian over 20 items, det at 1xTEa (h):", out[[c for c in out if c.startswith("det_h")]].median().round(2).to_dict())
print("low-sigma 5, MaxE(Nuf):\n", out.loc[low5, [c for c in out if c.startswith("maxenuf")]].round(1).to_string())
