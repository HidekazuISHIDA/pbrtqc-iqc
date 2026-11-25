"""Selection flow and per-analyte numbers (Methods, Table S1). Aggregates only.

Outputs: <tables>/methods_flow.csv (results remaining after each selection step), <tables>/methods_subjects.csv
"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import config as C, lis, spec, realsim as R, iqc_model as M

df = lis.results()
names = {lis.ITEM_NAME.get(i, i): i for i in spec.ITEMS}
x = df[df.item.astype(str).isin(list(names))].copy()
x["itm"] = x.item.astype(str).map(names)
f = {"全結果（20項目、2023–2025、全号機）": len(x)}
x = x[x.material.astype(str) == x.itm.map(lambda i: lis.MATERIAL.get(i, "血清"))]; f["基本検体（血清／GLUは血液）"] = len(x)
x = x[x.age >= 18]; f["18歳以上"] = len(x)
x = x[x.initial]; f["初回測定値"] = len(x)
x = x[x.value.notna()]; f["数値の結果"] = len(x)
x = x[x.unit == 1]; f["1号機"] = len(x)
for k, v in f.items():
    print(f"{k}: {v:,}")
rows = []
for it in spec.ITEMS:
    d = x[x.itm == it]
    rows.append(dict(item=it, analyzer=str(d.analyzer.iloc[0]), n_dev=int((d.date <= R.DEV[1]).sum()), n_val=int((d.date >= R.VAL[0]).sum()),
                     patients=d.pkey.nunique(), inpat=round(d.inpat.mean(), 2), qc_events=len(M.schedule(it, 1))))
pd.DataFrame(rows).to_csv(C.TABLES / "methods_subjects.csv", index=False)
pd.Series(f).to_csv(C.TABLES / "methods_flow.csv")
