"""Read the raw exports once and write the pseudonymised caches (<work>/processed/*.pkl).

Patient and specimen IDs are replaced by sequential keys; the original IDs are not written anywhere.
Prints only aggregate counts (files, rows, results per analyzer unit), never individual rows.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import config as C, lis, iqc_data

print(C.describe())
d = lis.build_cache()
print(f"LIS: {len(d):,} results, {d['sid'].nunique():,} specimens, {d['pkey'].nunique():,} patients; "
      f"{d['date'].min():%Y-%m-%d} to {d['date'].max():%Y-%m-%d}")
q = iqc_data.build_cache()
print(f"IQC: {len(q):,} QC results; materials {q['material'].value_counts().to_dict()}")
