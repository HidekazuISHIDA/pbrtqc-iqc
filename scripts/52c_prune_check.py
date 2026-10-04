"""Does the per-component pre-filter change the selected three-component settings? (internal review, 2026-10-04)

scripts/52 formed three-component combinations only from component settings whose own alarm rate met the limit. Because alarm
episodes less than 8 h apart are merged, adding a component can bridge two episodes, so a union can have fewer episodes than one
of its parts and the pre-filter is not guaranteed to be lossless. `52 --no-prune` repeats the selection with an exhaustive search
(same seed, same onsets). This script compares the two selections.
Reproducibility check: configurations without the pre-filter (two-component, inpatient + outpatient, pooled) and the candidate
table must be identical in both runs.
Output: outputs/tables/prune_check.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs/tables"

a = pd.read_csv(T / "select_pairs.csv")
b = pd.read_csv(T / "select_pairs_noprune.csv")
m = a.merge(b, on=["item", "cap", "system"], suffixes=("_pruned", "_exhaustive"))
m["same_setting"] = m["setting_pruned"] == m["setting_exhaustive"]
other = m[m["system"] != "aod+even+out"]
assert other["same_setting"].all(), other[~other["same_setting"]]                     # reproduction of the unaffected selections
da, db = pd.read_csv(T / "select_dev.csv"), pd.read_csv(T / "select_dev_noprune.csv")
assert da.equals(db) or np.allclose(da.select_dtypes("number"), db.select_dtypes("number"), equal_nan=True)
three = m[m["system"] == "aod+even+out"]
out = three[["item", "cap", "setting_pruned", "setting_exhaustive", "same_setting", "far_dev_wk_pruned", "far_dev_wk_exhaustive",
             "med_hours_dev_pruned", "med_hours_dev_exhaustive", "cap_met_pruned", "cap_met_exhaustive"]]
out.to_csv(T / "prune_check.csv", index=False)
print("reproduction of unaffected selections: OK (", len(other), "rows)")
print(out.groupby("cap")["same_setting"].agg(["sum", "count"]))
diff = out[~out["same_setting"]]
pd.set_option("display.width", 250)
print(diff.to_string(index=False) if len(diff) else "no differences")
