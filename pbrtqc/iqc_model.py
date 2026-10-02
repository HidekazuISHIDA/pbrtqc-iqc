"""IQC on the real QC data: events, lot segments, centre/SD (Total CV), Westgard z-scores (docs/14 §6, docs/18).

Event    QC passes of one analyzer unit within 1 h of each other form one QC event; the FIRST value of
         each level in the event is the decision value (later values are reruns).
Regular  the first event in the morning window (03:00-08:00) and the first in the midday window
         (08:30-15:00) of each day; all other events are "extra" (after calibration, reagent change or a
         rule violation - user 2026-09-30). Extra events are used as reset points, not for CV.
Lots     lot numbers are not in the extract (user: unknown). Lot changes are inferred per analyzer as the
         days on which many (item, level) series shift together by > 3 between-day SD
         (10-day median before vs after); a GA09II material switch (QAP -> Multiqual) also starts a segment.
Centre   mean of the regular QC in the first 20 days of each segment, then fixed (docs/18 §2).
SD       Total CV method (Murakoshi et al. 2024): the CV of regular QC within each segment of the
         development period, averaged over segments; SD = Total CV x centre.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import iqc_data as Q, iqc

DEV_END = pd.Timestamp("2024-12-31 23:59:59")
EVENT_GAP_H = 1.0
LOT_WIN = 10          # days before / after for the shift statistic
LOT_Z = 3.0
LOT_SHARE = 0.4       # share of series that must shift together
FIRST_DAYS = 20


def events(unit_df: pd.DataFrame) -> pd.Series:
    """Event id for every QC row of one analyzer unit (rows sorted by time)."""
    t = unit_df["time"].to_numpy()
    new = np.concatenate([[True], np.diff(t) > np.timedelta64(int(EVENT_GAP_H * 3600), "s")])
    return pd.Series(np.cumsum(new), index=unit_df.index)


def regular_flags(ev_time: pd.Series) -> np.ndarray:
    h = ev_time.dt.hour + ev_time.dt.minute / 60
    win = np.where((h >= 3) & (h < 8), 1, np.where((h >= 8.5) & (h < 15), 2, 0))
    df = pd.DataFrame({"day": ev_time.dt.normalize(), "win": win})
    rank = df.groupby(["day", "win"]).cumcount()
    return ((win > 0) & (rank == 0)).to_numpy()


def lot_changes(analyzer: str, unit: int, items: list[str]) -> list[pd.Timestamp]:
    """Inferred lot-change dates for one analyzer unit (see module docstring)."""
    days = pd.date_range("2023-01-01", "2025-12-31")
    Z = []
    for it in items:
        d = Q.load(it, unit, material="QAP")
        for lv in ("L", "H"):
            dm = d[d["level"] == lv].groupby(d["time"].dt.normalize())["value"].median().reindex(days)
            sd = dm.diff().abs().median() * 1.4826
            if not sd or np.isnan(sd):
                continue
            after = dm[::-1].rolling(LOT_WIN, min_periods=5).median()[::-1]
            before = dm.shift(1).rolling(LOT_WIN, min_periods=5).median()
            Z.append(((after - before) / sd).abs())
    Z = pd.concat(Z, axis=1)
    score = (Z > LOT_Z).sum(axis=1) / Z.shape[1]
    cand = score[score >= LOT_SHARE]
    out, block = [], []
    for day, v in cand.items():
        if block and (day - block[-1][0]).days > 5:
            out.append(max(block, key=lambda x: x[1])[0]); block = []
        block.append((day, v))
    if block:
        out.append(max(block, key=lambda x: x[1])[0])
    return out


def item_events(item: str, unit: int, lots: list[pd.Timestamp]) -> pd.DataFrame:
    """One row per QC event for one analyte: time, regular flag, segment, raw L/H values,
    centre, SD and z-scores for both levels (levels: QAP L/H or Multiqual 1/2 as L/H)."""
    all_unit = Q.results()
    d = Q.load(item, unit)
    an = d["analyzer"].iloc[0]
    u = all_unit[(all_unit["analyzer"] == an) & (all_unit["unit"] == unit)
                 & all_unit["material"].isin(["QAP", "Multiqual"])].sort_values("time")
    ev = events(u)
    d = d.merge(pd.DataFrame({"time": u["time"], "ev": ev}).drop_duplicates("time"), on="time", how="left")
    d["lvl"] = d["level"].astype(str).map({"L": "L", "H": "H", "1": "L", "2": "H"})
    first = d.sort_values("time").groupby(["ev", "lvl"]).first().reset_index()
    E = first.pivot(index="ev", columns="lvl", values="value")
    E["time"] = first.groupby("ev")["time"].min()
    E["material"] = first.groupby("ev")["material"].first().astype(str)
    E = E.sort_values("time").reset_index()
    E["regular"] = regular_flags(E["time"])
    # segments: inferred lot changes + material switches
    cuts = sorted(set(lots) | set(E.loc[E["material"].ne(E["material"].shift()), "time"].dt.normalize().iloc[1:]))
    E["seg"] = np.searchsorted(np.array(cuts, dtype="datetime64[ns]"), E["time"].to_numpy(), side="right")
    for lv in ("L", "H"):
        if lv not in E:
            E[lv] = np.nan
        cen = {}
        for s, g in E.groupby("seg"):
            first20 = g[(g["time"] < g["time"].min() + pd.Timedelta(days=FIRST_DAYS)) & g["regular"]][lv]
            cen[s] = first20.mean()
        E[f"c_{lv}"] = E["seg"].map(cen)
        dev = E[(E["time"] <= DEV_END) & E["regular"]]
        cvs = dev.groupby("seg")[lv].agg(lambda x: x.std(ddof=1) / x.mean() if x.count() >= 20 else np.nan)
        E.attrs[f"totalcv_{lv}"] = float(cvs.mean())
        E[f"sd_{lv}"] = E.attrs[f"totalcv_{lv}"] * E[f"c_{lv}"]
        E[f"z_{lv}"] = (E[lv] - E[f"c_{lv}"]) / E[f"sd_{lv}"]
    E.attrs["n_seg"] = int(E["seg"].nunique())
    return E


def rejections(E: pd.DataFrame, rules=iqc.RULE_SET_DEFAULT) -> np.ndarray:
    return iqc.evaluate_series(E[["z_L", "z_H"]].to_numpy(float), rules)


# --- robust Total CV and the QC schedule used by the fusion simulation ------------------------
# Lot numbers are unavailable and the raw QC contains level swaps and gross outliers, so the CV is
# estimated per calendar quarter after removing values > 4 robust SD from the quarter median, and the
# median over the quarters of the development period is taken (Total CV; lot changes affect at most
# one quarter each). Used for SD = CV x concentration in the QC simulation.

def total_cv(item: str, unit: int = 1) -> dict:
    d = Q.load(item, unit)
    d = d[d["time"] <= DEV_END].copy()
    d["lvl"] = d["level"].astype(str).map({"L": "L", "H": "H", "1": "L", "2": "H"})
    d["q"] = d["time"].dt.to_period("Q")
    out = {}
    for lv in ("L", "H"):
        per = []
        for _, g in d[d["lvl"] == lv].groupby(["q", "material"], observed=True):
            x = g["value"].to_numpy(float)
            med = np.median(x)
            mad = 1.4826 * np.median(np.abs(x - med))
            if mad > 0:
                x = x[np.abs(x - med) <= 4 * mad]
            if x.size >= 30:
                per.append(x.std(ddof=1) / x.mean())
        out[lv] = float(np.median(per))
    out["mean"] = (out["L"] + out["H"]) / 2
    return out


def schedule(item: str, unit: int = 1) -> pd.DataFrame:
    """Real QC event times (all events, regular + extra) for the analyzer running `item`."""
    all_unit = Q.results()
    an = Q.load(item, unit)["analyzer"].iloc[0]
    u = all_unit[(all_unit["analyzer"] == an) & (all_unit["unit"] == unit)
                 & all_unit["material"].isin(["QAP", "Multiqual"])].sort_values("time")
    t = u["time"].drop_duplicates().reset_index(drop=True)
    new = np.concatenate([[True], np.diff(t.to_numpy()) > np.timedelta64(int(EVENT_GAP_H * 3600), "s")])
    ev = t[new].reset_index(drop=True)
    return pd.DataFrame({"time": ev, "regular": regular_flags(ev)})
