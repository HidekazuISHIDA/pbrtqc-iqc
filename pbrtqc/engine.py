"""PBRTQC simulation engine.

Core ideas (see docs/03_PBRTQC方法論.md §3-§7):
  * a PBRTQC statistic is computed over patient results in measurement order,
    after truncation (values outside [L, U] are skipped, the statistic carries over);
  * control limits are set empirically on error-free baseline data so that the
    alarm-episode rate equals a target false-alarm rate (FAR);
  * performance = number of patient results from error onset until the first alarm
    (NPed), summarised as ANPed / MNPed over many random onsets.

All statistics take a 1-D float array and return an array of the same length
(NaN until the window is full; carried forward across truncated values).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import lfilter

# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------

def _expand(stat_incl: np.ndarray, incl: np.ndarray) -> np.ndarray:
    """Map a statistic computed on included values back onto the full series,
    carrying the last value forward over excluded positions."""
    out = np.full(incl.size, np.nan)
    out[incl] = stat_incl
    idx = np.where(incl, np.arange(incl.size), -1)
    idx = np.maximum.accumulate(idx)
    ok = idx >= 0
    out[ok] = out[idx[ok]]
    return out


def stat_ma(v, incl, N):
    x = v[incl]
    if x.size < N:
        return np.full(v.size, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    s = np.full(x.size, np.nan)
    s[N - 1:] = (c[N:] - c[:-N]) / N
    return _expand(s, incl)


def stat_ewma(v, incl, lam, warm=None):
    x = v[incl]
    if x.size == 0:
        return np.full(v.size, np.nan)
    z0 = x[: max(1, int(2 / lam))].mean()          # initialise at local mean
    s, _ = lfilter([lam], [1, -(1 - lam)], x, zi=[(1 - lam) * z0])
    w = warm if warm is not None else int(3 / lam)
    s[:min(w, s.size)] = np.nan
    return _expand(s, incl)


def stat_mm(v, incl, N):
    x = v[incl]
    s = pd.Series(x).rolling(N).median().to_numpy()
    return _expand(s, incl)


def stat_movsd(v, incl, N):
    x = v[incl]
    s = pd.Series(x).rolling(N).std(ddof=1).to_numpy()
    return _expand(s, incl)


def stat_movso(v, incl, N):
    """Moving sum of outliers = number of truncated (excluded) results in the last N results."""
    o = (~incl).astype(float)
    c = np.cumsum(np.insert(o, 0, 0.0))
    s = np.full(v.size, np.nan)
    if v.size >= N:
        s[N - 1:] = c[N:] - c[:-N]
    return s


ALGOS = {
    "MA":    dict(fn=stat_ma,    sided=2, pname="N"),
    "EWMA":  dict(fn=stat_ewma,  sided=2, pname="λ"),
    "MM":    dict(fn=stat_mm,    sided=2, pname="N"),
    "MovSD": dict(fn=stat_movsd, sided=1, pname="N"),
    "MovSO": dict(fn=stat_movso, sided=1, pname="N"),
}


def compute_stat(v: np.ndarray, algo: str, param, limits) -> np.ndarray:
    L, U = limits
    incl = np.ones(v.size, bool)
    if L is not None:
        incl &= v >= L
    if U is not None:
        incl &= v <= U
    return ALGOS[algo]["fn"](v, incl, param)


# ----------------------------------------------------------------------------
# truncation limits
# ----------------------------------------------------------------------------

def truncation_limits(v: np.ndarray, kind: str, ref=(None, None)):
    if kind == "none":
        return (None, None)
    if kind == "ref":
        return ref
    lo, hi = {"1-99": (1, 99), "2.5-97.5": (2.5, 97.5), "5-95": (5, 95), "10-90": (10, 90)}[kind]
    return tuple(np.nanpercentile(v, [lo, hi]))


# ----------------------------------------------------------------------------
# control limits from target false-alarm rate
# ----------------------------------------------------------------------------

def episodes(alarm: np.ndarray) -> int:
    a = alarm.astype(np.int8)
    return int(((a[1:] == 1) & (a[:-1] == 0)).sum() + (a[0] == 1))


def alarm_mask(stat, cl, sided):
    lo, hi = cl
    a = np.zeros(stat.size, bool)
    ok = ~np.isnan(stat)
    if hi is not None:
        a |= ok & (stat > hi)
    if lo is not None and sided == 2:
        a |= ok & (stat < lo)
    return a


def fit_control_limits(stat: np.ndarray, weeks: float, target_per_week: float, sided: int):
    """Find symmetric percentile limits such that alarm episodes/week ≈ target."""
    s = stat[~np.isnan(stat)]
    if s.size < 100:
        return (None, None), np.nan
    lo_p, hi_p = 1e-5, 0.2          # tail probability search range
    feasible = None                  # widest limits whose rate does not exceed target
    for _ in range(40):
        p = np.sqrt(lo_p * hi_p)
        cl = (np.quantile(s, p) if sided == 2 else None, np.quantile(s, 1 - p))
        rate = episodes(alarm_mask(stat, cl, sided)) / weeks
        if rate > target_per_week:
            hi_p = p
        else:
            lo_p = p
            if feasible is None or rate >= feasible[1]:
                feasible = (cl, rate)
        if hi_p / lo_p < 1.01:
            break
    if feasible is None:             # even the widest limits alarm too often (discrete statistic)
        cl = (np.quantile(s, lo_p) if sided == 2 else None, np.quantile(s, 1 - lo_p))
        feasible = (cl, episodes(alarm_mask(stat, cl, sided)) / weeks)
    return feasible


# ----------------------------------------------------------------------------
# error injection
# ----------------------------------------------------------------------------

def inject(seg: np.ndarray, kind: str, size: float, cva: float, rng, drift_len=500):
    """Apply an analytical error to a segment (all values after onset).
    size is expressed relative to concentration (e.g. k×TEa) for bias types,
    and as a multiplier of CVa for 'imprec' (2.0 → imprecision doubled)."""
    x = seg.astype(float).copy()
    if kind == "prop":
        x *= (1 + size)
    elif kind == "const":
        x += size * np.nanmedian(seg)
    elif kind == "drift":
        ramp = np.minimum(1.0, np.arange(x.size) / drift_len)
        x *= (1 + size * ramp)
    elif kind == "imprec":
        extra = np.sqrt(max(size ** 2 - 1, 0)) * cva * np.abs(x)
        x += rng.normal(0, 1, x.size) * extra
    else:
        raise ValueError(kind)
    return x


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------

def simulate(v: np.ndarray, t: np.ndarray, algo: str, param, trunc, cl, *, err_kind, err_size, cva,
             n_trials=100, warm=1500, horizon=2000, rng=None, return_example=False):
    """Random-onset error injection. Returns dict with NPed array (censored at horizon)."""
    rng = rng or np.random.default_rng(0)
    sided = ALGOS[algo]["sided"]
    n = v.size
    if n < warm + horizon + 10:
        return None
    starts = rng.integers(warm, n - horizon, n_trials)
    nped = np.empty(n_trials); det = np.zeros(n_trials, bool); minutes = np.full(n_trials, np.nan)
    example = None
    for k, i0 in enumerate(starts):
        seg = v[i0 - warm: i0 + horizon].copy()
        seg[warm:] = inject(seg[warm:], err_kind, err_size, cva, rng)
        st = compute_stat(seg, algo, param, trunc)
        a = alarm_mask(st, cl, sided)
        a[:warm] = False
        hit = np.flatnonzero(a)
        if hit.size:
            j = hit[0]; det[k] = True
            nped[k] = j - warm + 1
            minutes[k] = (t[i0 + (j - warm)] - t[i0]) / np.timedelta64(1, "m")
        else:
            nped[k] = horizon
        if return_example and example is None and hit.size:
            lo = max(0, warm - 400); hi = min(seg.size, hit[0] + 200)
            example = dict(value=seg[lo:hi].tolist(), stat=st[lo:hi].tolist(),
                           t=[str(x)[:16] for x in t[i0 - warm + lo: i0 - warm + hi]],
                           onset=int(warm - lo), alarm=int(hit[0] - lo))
    return dict(nped=nped, det=det, minutes=minutes, anped=float(nped.mean()),
                mnped=float(np.median(nped)), det_rate=float(det.mean()),
                med_minutes=float(np.nanmedian(minutes)) if det.any() else None, example=example)


def false_alarm_rate(stat, cl, sided, weeks):
    return episodes(alarm_mask(stat, cl, sided)) / weeks


# ----------------------------------------------------------------------------
# even check (Delta Plus-Minus Even Distribution Check)
# ----------------------------------------------------------------------------

def stat_rvalue(v: np.ndarray, prev_idx: np.ndarray, N: int) -> np.ndarray:
    """R value = (share of positive deltas among the last N eligible results) - 0.5.

    Only results whose same patient has a previous result inside the allowed gap take part
    (prev_idx >= 0); the statistic is carried forward over the others.
    Under a stable process R fluctuates around 0 with SE = 0.5/sqrt(N) (binomial).
    """
    ok = prev_idx >= 0
    idx = np.flatnonzero(ok)
    if idx.size < N:
        return np.full(v.size, np.nan)
    pos = (v[idx] - v[prev_idx[idx]] > 0).astype(float)
    tie = (v[idx] - v[prev_idx[idx]] == 0)
    pos[tie] = 0.5                                     # ties split evenly
    c = np.cumsum(np.insert(pos, 0, 0.0))
    r = np.full(idx.size, np.nan)
    r[N - 1:] = (c[N:] - c[:-N]) / N - 0.5
    return _expand(r, ok)


def rvalue_limits(N: int, far_per_week: float, per_week: float) -> tuple:
    """Binomial control limits for the R value at a target alarm rate.

    R has SE = 0.5/sqrt(N) under the null, so the limits are +/- z * SE where z comes from
    the number of independent opportunities per week.
    """
    from scipy.stats import norm
    se = 0.5 / np.sqrt(N)
    opportunities = max(per_week / N, 1.0)             # non-overlapping blocks per week
    p = np.clip(far_per_week / opportunities, 1e-9, 0.5)
    z = norm.isf(p / 2)
    return (-z * se, z * se)


ALGOS["EVEN"] = dict(fn=None, sided=2, pname="N")


def simulate_even(v, t, prev_idx, N, cl, *, err_kind, err_size, cva, n_trials=100,
                  warm=1500, horizon=2000, rng=None):
    """Random-onset error injection for even check.

    The previous value of a patient measured before the onset is unaffected, so the delta
    is shifted; once the error has lasted longer than the typical repeat interval both
    values carry it and the signal fades. That behaviour is reproduced here because the
    deltas are recomputed from the modified series.
    """
    rng = rng or np.random.default_rng(0)
    n = v.size
    if n < warm + horizon + 10:
        return None
    starts = rng.integers(warm, n - horizon, n_trials)
    nped = np.empty(n_trials); det = np.zeros(n_trials, bool); minutes = np.full(n_trials, np.nan)
    for k, i0 in enumerate(starts):
        a0, a1 = i0 - warm, i0 + horizon
        seg = v[a0:a1].copy()
        seg[warm:] = inject(seg[warm:], err_kind, err_size, cva, rng)
        p = prev_idx[a0:a1] - a0                        # remap; anything before the window drops out
        p[prev_idx[a0:a1] < 0] = -1
        p[p < 0] = -1
        st = stat_rvalue(seg, p, N)
        al = alarm_mask(st, cl, 2)
        al[:warm] = False
        hit = np.flatnonzero(al)
        if hit.size:
            j = hit[0]; det[k] = True; nped[k] = j - warm + 1
            minutes[k] = (t[i0 + (j - warm)] - t[i0]) / np.timedelta64(1, "m")
        else:
            nped[k] = horizon
    return dict(nped=nped, det=det, minutes=minutes, anped=float(nped.mean()),
                mnped=float(np.median(nped)), det_rate=float(det.mean()),
                med_minutes=float(np.nanmedian(minutes)) if det.any() else None, example=None)
