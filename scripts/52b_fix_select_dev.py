"""Repair select_dev*.csv written before 2026-10-04: the AoD rows lacked the lookback G in "trunc".

scripts/52 loops G -> truncation -> N for AoD, so within each item the 24 AoD rows are in that order. This script assigns
"trunc|G" by that order and verifies every row by recomputing its development-period alarm rate (deterministic, no
simulation) from the same data. Aborts on any mismatch. scripts/52 itself writes "trunc|G" from now on.
Usage: python3 scripts/52b_fix_select_dev.py [final]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import lis, spec, realsim as R

SUF = "_final" if "final" in sys.argv[1:] else ""
if SUF:
    lis.VALUE = "final"
p = ROOT / f"outputs/tables/select_dev{SUF}.csv"
sd = pd.read_csv(p)
if sd.loc[sd["stream"] == "aod", "trunc"].str.contains(r"\|G").all():
    sys.exit(f"{p.name}: already repaired")
order = [(G, tr, N) for G in R.AOD_LOOKBACK for tr in R.AOD_TRUNCS for N in R.NS]
for it in spec.ITEMS:
    idx = sd.index[(sd["item"] == it) & (sd["stream"] == "aod")]
    assert len(idx) == len(order), (it, len(idx))
    d_all = lis.load(it, unit=None, start=R.DEV[0], end=R.VAL[1])
    tabs = {G: R.delta_table(d_all, unit=1, lookback=G) for G in R.AOD_LOOKBACK}
    for i, (G, tr, N) in zip(idx, order):
        far = R.rate_per_week(R.episodes(R.make_aod_stream(tabs[G], tr, N)), R.DEV)
        assert sd.at[i, "trunc"] == tr and sd.at[i, "N"] == N and np.isclose(sd.at[i, "far_dev_wk"], far), (it, G, tr, N)
        sd.at[i, "trunc"] = f"{tr}|G{G}"
    print(it, "verified", flush=True)
sd.to_csv(p, index=False)
print("repaired", p.name)
