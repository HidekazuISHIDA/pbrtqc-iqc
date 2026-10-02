"""Real-data PBRTQC helpers: per-stream limits, alarm episodes, and error injection (docs/15).

A stream is one analyte on one analyzer restricted to inpatients, outpatients or all (pooled),
in arrival-time order. Limits always come from the development period.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import engine as E, spec

DEV = ("2023-01-01", "2024-12-31")
VAL = ("2025-01-01", "2025-12-31")
TRUNCS = ["ref", "2.5-97.5", "none"]
NS = [20, 50, 100, 200]
MERGE_H = 8.0
FOLLOW = np.timedelta64(7, "D")
WARM = 1000
TEA_PROVISIONAL = {"Mg": 0.0385}   # EFLM BV DB desirable, confirmed by the user 2026-10-01 (HANDOVER A-10)


def tea(item: str, S=None) -> float:
    S = S or spec.load()
    return S[item]["tea"] or TEA_PROVISIONAL[item]


@dataclass
class Stream:
    v: np.ndarray          # values
    t: np.ndarray          # arrival times (sorted)
    dev: np.ndarray        # bool, development period
    lim: tuple             # truncation limits
    cl: tuple              # control limits
    stat: np.ndarray       # statistic on the whole series
    N: int


def make_stream(d: pd.DataFrame, mask: np.ndarray, item: str, trunc: str, N: int) -> Stream | None:
    """None when the setting is undefined (ref truncation for an item without a common RI)."""
    v = d.loc[mask, "value"].to_numpy(float)
    t = d.loc[mask, "arrival"].to_numpy()
    dev = (d.loc[mask, "date"] <= DEV[1]).to_numpy()
    if trunc == "ref":                       # JCCLS common reference interval, sex-specific where defined
        lim = spec.ref_limits(item, d.loc[mask, "sex"].to_numpy())
        if lim is None:
            return None
    else:
        lim = E.truncation_limits(v[dev], trunc)
    stat = E.compute_stat(v, "MA", N, lim)
    sd = stat[dev]
    mu, sg = np.nanmean(sd), np.nanstd(sd)
    return Stream(v, t, dev, lim, (mu - 3 * sg, mu + 3 * sg), stat, N)


def stream_masks(d: pd.DataFrame) -> dict:
    inp = d["inpat"].to_numpy()
    return {"in": inp, "out": ~inp, "pooled": np.ones(len(d), bool)}


# --- alarm episodes -----------------------------------------------------------------

def intervals(alarm: np.ndarray, t: np.ndarray) -> list:
    a = alarm.astype(np.int8)
    dd = np.diff(np.concatenate([[0], a, [0]]))
    s, e = np.where(dd == 1)[0], np.where(dd == -1)[0] - 1
    return list(zip(t[s], t[e]))


def merge(iv: list, gap_h=MERGE_H) -> list:
    out = []
    gap = np.timedelta64(int(gap_h * 3600), "s")
    for s, e in sorted(iv):
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def episodes(st: Stream) -> list:
    return merge(intervals(E.alarm_mask(st.stat, st.cl, 2), st.t))


def rate_per_week(iv: list, period) -> float:
    a, b = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)
    n = sum(1 for s, _ in iv if np.datetime64(a) <= s < np.datetime64(b))
    return n / ((b - a).days / 7)


# --- error injection ----------------------------------------------------------------

def first_alarm_hours(st: Stream, t_on, bias: float):
    """(results in this stream from onset to alarm, hours to alarm) or (inf, inf)."""
    i0 = int(np.searchsorted(st.t, t_on))
    i_end = int(np.searchsorted(st.t, t_on + FOLLOW))
    if i0 >= i_end:
        return np.inf, np.inf
    lo = max(0, i0 - WARM)
    seg = st.v[lo:i_end].copy()
    seg[i0 - lo:] *= (1 + bias)
    lim = tuple(x[lo:i_end] if isinstance(x, np.ndarray) else x for x in st.lim)
    a = E.alarm_mask(E.compute_stat(seg, "MA", st.N, lim), st.cl, 2)[i0 - lo:]
    hit = np.flatnonzero(a)
    if hit.size == 0:
        return np.inf, np.inf
    return float(hit[0] + 1), (st.t[i0 + hit[0]] - t_on) / np.timedelta64(1, "h")


def onsets(d: pd.DataFrame, period, n: int, rng) -> np.ndarray:
    """Onset times drawn uniformly over results in `period` (leaving 7 days of follow-up)."""
    t = d["arrival"].to_numpy()
    ok = (d["date"] >= period[0]) & (d["date"] <= pd.Timestamp(period[1]) - pd.Timedelta(days=7))
    return t[rng.choice(np.flatnonzero(ok.to_numpy()), n)]


def summarise(nh: np.ndarray) -> dict:
    """nh: array (trials, 2) of (results, hours); inf = not detected within follow-up."""
    return dict(det_rate=float(np.isfinite(nh[:, 1]).mean()), mnped=float(np.median(nh[:, 0])),
                med_hours=float(np.median(nh[:, 1])), p_det_24h=float((nh[:, 1] <= 24).mean()))


# --- average of patient deltas (AoD) --------------------------------------------------
# Cembrowski GS, Xu Q, Cervinski MA. Clin Chem 2021;67:1019-29 (PMID 33993233).
# Stream = results on the monitored unit whose patient has a previous result (any unit) 6 h - G days
# earlier (G = lookback setting); statistic = moving average of (current - previous). A persistent error on the monitored
# unit shifts current values; it cancels only when the previous value is also on that unit after onset.

AOD_TRUNCS = ["2.5-97.5", "none"]
AOD_MIN_GAP = 0.25             # days (previous result more than 6 h earlier)
AOD_LOOKBACK = [7, 30, 90]  # days: how far back the previous result may be (setting G; max 90 d, user 2026-09-29)


@dataclass
class AoDStream:
    cur: np.ndarray
    prev: np.ndarray
    prev_u: np.ndarray         # previous value measured on the monitored unit
    t: np.ndarray              # time of the current result
    t_prev: np.ndarray
    dev: np.ndarray
    lim: tuple
    cl: tuple
    stat: np.ndarray
    N: int
    kind: str = "aod"          # "aod" = mean of deltas; "even" = even check R value (sign of deltas)


def _transform(dl: np.ndarray, kind: str) -> np.ndarray:
    """even check (Hatanaka N et al. J Appl Lab Med 2024;9:316, PMID 38170846): score 1 / 0.5 / 0 for a
    positive / tied / negative delta; the moving average of the score minus 0.5 is the R value."""
    if kind == "aod":
        return dl
    sc = np.where(dl > 0, 1.0, np.where(dl < 0, 0.0, 0.5))
    sc[np.isnan(dl)] = np.nan
    return sc - 0.5


def delta_table(d_all: pd.DataFrame, unit: int = 1, lookback: float = 7.0) -> pd.DataFrame:
    """Rows of the monitored unit with their patient's previous result (any unit) 6 h - `lookback` days earlier."""
    a = d_all.sort_values(["arrival", "acc"]).reset_index(drop=True)
    g = a.groupby("pkey")
    a["prev"], a["t_prev"], a["prev_unit"] = g["value"].shift(1), g["arrival"].shift(1), g["unit"].shift(1)
    gap = (a["arrival"] - a["t_prev"]).dt.total_seconds() / 86400
    ok = (a["unit"] == unit) & (gap > AOD_MIN_GAP) & (gap <= lookback)
    return a[ok].reset_index(drop=True)


def make_aod_stream(dt: pd.DataFrame, trunc: str, N: int, unit: int = 1, kind: str = "aod") -> AoDStream:
    """kind="even" gives even check (no truncation: the score is bounded)."""
    cur, prev = dt["value"].to_numpy(float), dt["prev"].to_numpy(float)
    dev = (dt["date"] <= DEV[1]).to_numpy()
    dl = _transform(cur - prev, kind)
    lim = (None, None) if kind == "even" else E.truncation_limits(dl[dev], trunc)
    stat = E.compute_stat(dl, "MA", N, lim)
    sd = stat[dev]
    mu, sg = np.nanmean(sd), np.nanstd(sd)
    return AoDStream(cur, prev, (dt["prev_unit"] == unit).fillna(False).to_numpy(bool), dt["arrival"].to_numpy(),
                     dt["t_prev"].to_numpy(), dev, lim, (mu - 3 * sg, mu + 3 * sg), stat, N, kind)


def _first_alarm_aod(st: AoDStream, t_on, bias: float):
    i0 = int(np.searchsorted(st.t, t_on))
    i_end = int(np.searchsorted(st.t, t_on + FOLLOW))
    if i0 >= i_end:
        return np.inf, np.inf
    lo = max(0, i0 - WARM)
    cur = st.cur[lo:i_end].copy()
    prev = st.prev[lo:i_end].copy()
    cur[i0 - lo:] *= (1 + bias)
    hit_prev = st.prev_u[lo:i_end] & (st.t_prev[lo:i_end] >= t_on)
    prev[hit_prev] *= (1 + bias)
    a = E.alarm_mask(E.compute_stat(_transform(cur - prev, st.kind), "MA", st.N, st.lim), st.cl, 2)[i0 - lo:]
    hit = np.flatnonzero(a)
    if hit.size == 0:
        return np.inf, np.inf
    return float(hit[0] + 1), (st.t[i0 + hit[0]] - t_on) / np.timedelta64(1, "h")


_first_alarm_ma = first_alarm_hours


def first_alarm_hours(st, t_on, bias: float):  # noqa: F811  (dispatch on stream type)
    if isinstance(st, AoDStream):
        return _first_alarm_aod(st, t_on, bias)
    return _first_alarm_ma(st, t_on, bias)
