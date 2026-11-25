"""Synthetic raw exports in the same column layout as the laboratory's LIS and IQC exports (no patient data).

Lets the whole pipeline run end to end without real data (profile "demo", pbrtqc/config.py). Everything is
generated from random numbers with a fixed seed, so the files - and every result derived from them - are reproducible.

LIS export (one row per measurement pass of a specimen; columns as in pbrtqc/lis.py):
  受付日, PatientID, 依頼元, 到着時刻, HISDATE, 年齢, 入外区分, 依頼科名, 性別, then groups of (項目名, 材料, 検査値, 装置名)
  - inpatients: morning ward draws on most days of their stay; outpatients: daytime clinic visits with revisits;
    evening/emergency samples. Inpatients have lower set points for proteins etc. (the daily case-mix problem).
  - per patient and analyte: a set point (between-subject variation), within-subject and analytical variation.
  - two analyzer units per type; specimens split between them.
  - serum indices (hemolysis, lipemia, icterus) on every specimen row, as in the export.
  - reruns: about 12 % of specimens have an adjacent second row (1-3 analytes, half on the other unit);
    a few first results are non-numeric ("要希釈再検"); a few minors and non-serum specimens (filtered by the loader).
IQC export (one row per QC pass; columns as in pbrtqc/iqc_data.py): two levels (QAP L/H; GA09II also Multiqual 1/2 from
  2024-03 and only Multiqual from 2025-09, as in the laboratory) per analyzer unit at about
  04:30 every day and 10:30 on weekdays, occasional extra events and reruns, and QC-material lot changes every ~4 months.
Injected analytical error (to show detection in the monitor and the shadow analysis): Ca on BM8040 #1,
  +1.5 x TEa from 2025-10-01 08:00 to 2025-10-03 12:00, in patient results and QC.

Usage: python3 scripts/00_make_synthetic_raw.py      (writes to work_demo/raw/, about 1-2 min)
"""
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbrtqc import spec

OUT = ROOT / "work_demo" / "raw"
rng = np.random.default_rng(20261125)
DAYS = pd.date_range("2023-01-01", "2025-12-31")
ERR = dict(item="Ca", analyzer="BM8040", unit=1, start=pd.Timestamp("2025-10-01 08:00"), end=pd.Timestamp("2025-10-03 12:00"), k=1.5)

# key: (LIS name, analyzer, median, between-subject CV, within-subject CV, analytical CV, inpatient shift,
#       P(ordered | inpatient), P(ordered | outpatient), QC L, QC H)   - generic illustrative values
A = {
    "Na":    ("Na", "EA10M", 140, .015, .006, .006, 0.00, .97, .93, 130, 150),
    "K":     ("K", "EA10M", 4.2, .08, .05, .007, -0.03, .97, .93, 4.0, 6.0),
    "Cl":    ("Cl", "EA10M", 104, .02, .012, .006, -0.01, .97, .93, 95, 115),
    "Ca":    ("Ca", "BM8040", 9.0, .04, .02, .012, -0.05, .65, .55, 7.5, 10.5),
    "TP":    ("TP", "BM8040", 6.8, .07, .03, .007, -0.12, .85, .85, 5.0, 7.5),
    "ALB":   ("ALB", "BM8040", 4.0, .10, .03, .015, -0.20, .95, .92, 3.0, 4.5),
    "UN":    ("UN", "BM8040", 14, .30, .12, .011, 0.15, .97, .93, 15, 50),
    "CRE":   ("CRE", "BM8040", 0.8, .25, .06, .008, 0.00, .97, .95, 1.0, 4.4),
    "UA":    ("UA", "BM8040", 5.3, .25, .08, .007, -0.10, .45, .60, 3.5, 8.0),
    "AST":   ("AST", "BM8040", 21, .35, .12, .010, 0.05, .95, .93, 35, 115),
    "ALT":   ("ALT", "BM8040", 18, .50, .15, .014, 0.05, .95, .93, 30, 120),
    "LD":    ("LD", "BM8040", 190, .18, .07, .008, 0.05, .92, .88, 150, 400),
    "ALP":   ("ALP･IFCC", "BM8040", 80, .30, .07, .013, 0.10, .88, .85, 100, 350),
    "γ-GT":  ("γ-GT", "BM8040", 25, .70, .12, .016, 0.10, .88, .88, 40, 150),
    "CK":    ("CK", "BM8040", 100, .55, .25, .013, -0.20, .80, .78, 100, 400),
    "AMY":   ("AMY", "BM8040", 75, .30, .09, .006, -0.05, .55, .60, 60, 300),
    "GLU":   ("GLU", "GA09Ⅱ", 108, .15, .10, .010, -0.03, .70, .80, 80, 250),
    "TG":    ("TG", "BM8040", 110, .50, .20, .008, 0.00, .15, .45, 80, 200),
    "T-CHO": ("T-CHO", "BM8040", 190, .18, .06, .006, -0.20, .10, .35, 150, 250),
    "Mg":    ("Mg", "BM8040", 2.0, .08, .04, .016, 0.00, .30, .12, 1.5, 4.5),
}
ITEMS = list(A)
assert set(ITEMS) == set(spec.ITEMS)
DEC = {it: spec.DECIMALS.get(it, 0) for it in ITEMS}
WARDS = ["3階東病棟", "4階西病棟", "5階東病棟", "6階西病棟", "ICU", "HCU"]
CLINICS = ["内科外来", "外科外来", "腎臓内科外来", "消化器内科外来", "整形外科外来"]
DEPTS = ["内科", "外科", "腎臓内科", "消化器内科", "整形外科", "循環器内科"]


def fmt(v, d):
    return f"{v:.{d}f}" if d else str(int(round(v)))


# ---------------------------------------------------------------- specimens
rows = []          # (patient, inpat, arrival)
next_pid = 1
inpatients = []    # [pid, discharge day]
out_pool = np.arange(1_000_000, 1_008_000)          # outpatient ids, revisits drawn at random
for day in DAYS:
    wk = day.dayofweek < 5
    inpatients = [p for p in inpatients if p[1] > day]
    while len(inpatients) < 100:
        inpatients.append([next_pid, day + pd.Timedelta(days=int(rng.integers(3, 21)))])
        next_pid += 1
    for p, _ in inpatients:
        if rng.random() < (0.7 if wk else 0.35):
            rows.append((p, True, day + pd.Timedelta(minutes=float(rng.uniform(300, 480)))))
    for p in rng.choice(out_pool, int(rng.poisson(120 if wk else 5)), replace=False):
        rows.append((int(p), False, day + pd.Timedelta(minutes=float(np.clip(rng.normal(630, 95), 500, 990)))))
    for _ in range(int(rng.poisson(12))):                                 # evening / emergency
        inp = rng.random() < 0.6
        pid = next_pid if inp else int(rng.choice(out_pool))
        next_pid += inp
        rows.append((pid, inp, day + pd.Timedelta(minutes=float(rng.uniform(960, 1435)))))
sp = pd.DataFrame(rows, columns=["pid", "inpat", "arrival"]).sort_values("arrival", kind="stable").reset_index(drop=True)
sp["arrival"] = sp["arrival"].dt.floor("s")
n = len(sp)
pids = sp["pid"].unique()
prof = pd.DataFrame({"pid": pids})
prof["age"] = np.where(rng.random(pids.size) < 0.04, rng.integers(0, 18, pids.size),
                       np.clip(rng.normal(66, 15, pids.size), 18, 99)).astype(int)
prof["sex"] = np.where(rng.random(pids.size) < 0.5, "男性", "女性")
prof = prof.set_index("pid")
sp["age"] = prof.loc[sp["pid"], "age"].to_numpy()
sp["sex"] = prof.loc[sp["pid"], "sex"].to_numpy()
sp["source"] = np.where(sp["inpat"], rng.choice(WARDS, n), rng.choice(CLINICS, n))
sp["dept"] = rng.choice(DEPTS, n)
sp["inout"] = np.where(sp["inpat"], "入院", "外来")
sp["his"] = [f"S{i:08d}" for i in range(1, n + 1)]
pidx = pd.Series(np.arange(pids.size), index=pids)[sp["pid"]].to_numpy()

# ---------------------------------------------------------------- results per analyte
res = {}
for it, (_, an, med, cvg, cvi, cva, shift, p_in, p_out, *_q) in A.items():
    ordered = rng.random(n) < np.where(sp["inpat"], p_in, p_out)
    setp = med * np.exp(rng.normal(0, cvg, pids.size))
    v = setp[pidx] * np.where(sp["inpat"], 1 + shift, 1.0) * (1 + rng.normal(0, cvi, n)) * (1 + rng.normal(0, cva, n))
    unit = np.where(rng.random(n) < 0.5, 1, 2)
    if it == ERR["item"]:
        hit = (unit == ERR["unit"]) & (sp["arrival"] >= ERR["start"]).to_numpy() & (sp["arrival"] < ERR["end"]).to_numpy()
        tea_it = (spec.JSCC[it][1] + 1.65 * spec.JSCC[it][0]) / 100           # TEa = BA + 1.65 CVA (JSCC)
        v = np.where(hit, v * (1 + ERR["k"] * tea_it), v)
    res[it] = dict(ordered=ordered, value=v, unit=unit, an=an, cva=cva)

# ---------------------------------------------------------------- write LIS export (half-year files)
OUT.joinpath("lis").mkdir(parents=True, exist_ok=True)
hdr = ["受付日", "PatientID", "依頼元", "到着時刻", "HISDATE", "年齢", "入外区分", "依頼科名", "性別"] + \
      ["項目名", "材料", "検査値", "装置名"] * 24
rerun = rng.random(n) < 0.12
files = {}
for half in sorted({(t.year, 1 if t.month <= 6 else 7) for t in DAYS}):
    files[half] = open(OUT / "lis" / f"lis_{half[0]}{half[1]:02d}.csv", "w", encoding="utf-8", newline="")
    csv.writer(files[half]).writerow(hdr)
writers = {k: csv.writer(f) for k, f in files.items()}
for i, s in enumerate(sp.itertuples(index=False)):
    t = s.arrival
    w = writers[(t.year, 1 if t.month <= 6 else 7)]
    age = f"{s.age}歳" if s.age >= 1 else f"{int(rng.integers(1, 12))}ヶ月"
    head = [t.strftime("%Y/%m/%d"), f"DEMO{s.pid:07d}", s.source, t.strftime("%Y/%m/%d %H:%M:%S"), s.his, age, s.inout, s.dept, s.sex]
    groups, rr = [], []
    other_mat = rng.random() < 0.003
    for it in ITEMS:
        r = res[it]
        if not r["ordered"][i]:
            continue
        name = A[it][0]
        mat = "血液" if it == "GLU" else ("胸水" if other_mat else "血清")
        val = fmt(r["value"][i], DEC[it])
        if it == "CK" and r["value"][i] > 2500:
            val = "要希釈再検"
        groups += [name, mat, val, f"{r['an']} #{r['unit'][i]}"]
        if val == "要希釈再検" or (rerun[i] and rng.random() < 0.15):
            u2 = r["unit"][i] if rng.random() < 0.5 else 3 - r["unit"][i]
            v2 = r["value"][i] * (1 + rng.normal(0, r["cva"]))
            rr += [name, mat, fmt(v2, DEC[it]), f"{r['an']} #{u2}"]
    if not groups:
        continue
    u_bm = next((res[it]["unit"][i] for it in ITEMS if A[it][1] == "BM8040" and res[it]["ordered"][i]), 1)
    for ix in ("検体溶血", "検体乳ビ", "検体黄疸"):            # serum indices, reported with the chemistry results
        groups += [ix, "血清", "1+" if rng.random() < 0.03 else "-", f"BM8040 #{u_bm}"]
    w.writerow(head + groups)
    if rr:                                                    # adjacent rerun row, same specimen and arrival time
        w.writerow(head + rr)
for f in files.values():
    f.close()

# ---------------------------------------------------------------- write IQC export
OUT.joinpath("qc").mkdir(parents=True, exist_ok=True)
qhdr = ["受付日", "受付番号", "患者ID", "カナ", "依頼元", "医師", "到着時刻", "HISコード", "年齢", "入外区分", "依頼科名", "性別"] + \
       ["項目名", "材料", "検査値", "装置名"] * 16
by_an = {}
for it, a in A.items():
    by_an.setdefault(a[1], []).append(it)
lots = {an: sorted(rng.choice(DAYS[30:-30], 9, replace=False)) for an in by_an}       # lot changes, about every 4 months
with open(OUT / "qc" / "qc_2023_2025.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(qhdr)
    seq = 0
    for an, items in by_an.items():
        for u in (1, 2):
            target = {(it, lv): A[it][9 + j] for it in items for j, lv in enumerate("LH")}
            lot_i = 0
            for day in DAYS:
                while lot_i < len(lots[an]) and day >= lots[an][lot_i]:
                    for key in target:
                        target[key] *= 1 + rng.normal(0, 1.5 * A[key[0]][5])
                    lot_i += 1
                times = [day + pd.Timedelta(minutes=float(rng.normal(270, 15)))]
                if day.dayofweek < 5:
                    times.append(day + pd.Timedelta(minutes=float(rng.normal(630, 20))))
                if rng.random() < 0.15:
                    times.append(day + pd.Timedelta(minutes=float(rng.uniform(900, 1380))))
                for t0 in times:
                    mats = []                         # GA09II: QAP until 2025-08, Multiqual from 2024-03 (as in the laboratory)
                    if an != "GA09Ⅱ" or t0 < pd.Timestamp("2025-09-01"):
                        mats += [("QAP", "L", 0), ("QAP", "H", 1)]
                    if an == "GA09Ⅱ" and t0 >= pd.Timestamp("2024-03-01"):
                        mats += [("Multiqual", "L", 2), ("Multiqual", "H", 3)]
                    passes = mats + ([mats[0][:2] + (4,)] if rng.random() < 0.05 else [])
                    for mat, lv, dm in passes:
                        t = (t0 + pd.Timedelta(minutes=dm)).floor("s")
                        g = []
                        for it in items:
                            x = target[(it, lv)] * (1.03 if mat == "Multiqual" else 1.0) * (1 + rng.normal(0, A[it][5]))
                            if it == ERR["item"] and an == ERR["analyzer"] and u == ERR["unit"] and ERR["start"] <= t < ERR["end"]:
                                x *= 1 + ERR["k"] * (spec.JSCC[it][1] + 1.65 * spec.JSCC[it][0]) / 100
                            g += [A[it][0], "その他", fmt(x, DEC[it] + 1), f"{an} #{u}"]
                        seq += 1
                        kana = f"QAP {lv}" if mat == "QAP" else f"Multiqual {'1' if lv == 'L' else '2'}"
                        w.writerow([t.strftime("%Y/%m/%d"), str(seq), f"QC-{an}-{u}-{mat}-{lv}", kana, "検査部", "", t.strftime("%Y/%m/%d %H:%M:%S"),
                                    "", "", "外来", "検査部", ""] + g)
print("specimens", n, "| files written to", OUT)
