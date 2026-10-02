"""Parvin-style patient-risk model: E(Nu), E(Nuf), MaxE(Nuf), and the combined IQC+PBRTQC case.

Notation (docs/05_IQC融合.md §3, docs/06_シャドー運用設計.md I-1/I-7)
  N        run size: patient results between consecutive QC events
  Ped(SE)  probability the QC rule rejects at the first QC event after an error of size SE
  P_E(SE)  probability a reported result exceeds TEa under a systematic error SE
  ΔP_E(SE) P_E(SE) − P_E(0): the increase caused by the out-of-control condition

All error sizes are in stable analytical SD units, so TEa enters as the sigma metric
    tea_sd = (TEa% − |bias%|) / CVa%.

Two quantities, deliberately kept separate:

E(Nu)  — unreliable results *produced* before detection. Assumption-light.
         Distance from onset to detection by IQC alone:
             d_iqc = N × (1/Ped − 0.5)
         (on average N/2 results remain in the run where the error started, then
          (1−Ped)/Ped further complete runs are needed before a QC event rejects)
             E(Nu) = ΔP_E × d_iqc

E(Nuf) — unreliable results that become *final*, i.e. cannot be recalled. The laboratory
         repeats back to the last accepted QC event, so the results of the detecting run
         are recovered and only results released in earlier runs are final. Writing
         q = 1 − Ped and summing over the run in which detection occurs:
             E(Nuf) = ΔP_E × N × [ q/2 + q²/Ped ]
         This tends to 0 as Ped → 1: a gross error is caught at the very next QC event and
         everything since the last accepted event is recalled. The maximum over SE is
         therefore attained at intermediate error sizes — MaxE(Nuf), whose usual goal is < 1.

Combined IQC + shadow PBRTQC: PBRTQC alarms about `nped` results after onset, wherever the
QC events happen to fall, and the recall still reaches back to the last accepted QC event.
With the onset uniform inside the run, results become final only if the error crossed a QC
event that accepted it — which happens with probability q = 1 − Ped — so for nped ≤ N

    E(Nuf)_combined = ΔP_E × q × nped² / (2N)

and for nped > N the crossing is certain, giving ΔP_E × q × (nped − N/2). Adding a second
detector can never make matters worse, so the result is capped at the IQC-only value.

Verification (2026-09-29, HANDOVER A-2): the IQC-only E(Nuf) above is algebraically identical to
Parvin's E(NU) for a continuous-mode process with bracketed QC and a persistent error
(Parvin CA. Clin Chem 2008;54:2049-54, PMID 18927244):
    E(NU) = ΔP_E {(ARL_ED − 1) E(N_B) − (1 − P_1)[E(N_B) − E(N_0)]}
with ARL_ED = 1/Ped, P_1 = Ped, E(N_0) = N/2  →  ΔP_E [ q E(N_0) + (q²/Ped) E(N_B) ].
"Recall back to the last accepted QC event" and "hold results until the next QC passes" leave
the same results final. Checked numerically (max relative difference 2e-13).
The same form shows how to treat unequal runs: the run in which the error starts is
length-biased (E(N_0) = N_eff/2, N_eff = E[N²]/E[N]) while later runs have the plain mean
E(N_B) = E[N]. Pass both via `n_onset` (see effective_run_size).
The combined IQC + PBRTQC expression is NOT in Parvin's papers; the primary analysis counts
final results directly in the coupled simulation instead (HANDOVER B-3).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from . import iqc


def p_exceed(tea_sd: float, se_sd: float = 0.0, re_mult: float = 1.0) -> float:
    """P(|measurement error| > TEa) under a systematic shift se_sd and SD multiplier re_mult."""
    hi = (tea_sd - se_sd) / re_mult
    lo = (-tea_sd - se_sd) / re_mult
    return float(norm.sf(hi) + norm.cdf(lo))


def delta_pe(tea_sd: float, se_sd: float, re_mult: float = 1.0) -> float:
    return max(0.0, p_exceed(tea_sd, se_sd, re_mult) - p_exceed(tea_sd, 0.0, 1.0))


def distance_iqc(run_size: float, ped: float, n_onset: float | None = None) -> float:
    """Expected patient results from error onset to detection by IQC alone.

    run_size = mean run E[N]; n_onset = size of the run containing the onset (N_eff for
    unequal runs, defaults to run_size)."""
    n0 = run_size if n_onset is None else n_onset
    return n0 / 2.0 + run_size * (1.0 / max(ped, 1e-6) - 1.0)


def e_nu(tea_sd: float, se_sd: float, run_size: float, ped: float,
         n_onset: float | None = None) -> float:
    return delta_pe(tea_sd, se_sd) * distance_iqc(run_size, ped, n_onset)


def e_nuf(tea_sd: float, se_sd: float, run_size: float, ped: float,
          n_onset: float | None = None) -> float:
    """Parvin 2008 E(NU), bracketed/recall, persistent error. See module docstring."""
    q = 1.0 - min(max(ped, 1e-6), 1.0)
    n0 = run_size if n_onset is None else n_onset
    return delta_pe(tea_sd, se_sd) * (q * n0 / 2.0 + q * q / max(ped, 1e-6) * run_size)


def _final_distance_pbrtqc(nped: float, run_size: float, ped: float) -> float:
    """Post-onset results already released past an *accepting* QC event when PBRTQC alarms."""
    q = 1.0 - min(max(ped, 0.0), 1.0)
    d = nped ** 2 / (2.0 * run_size) if nped <= run_size else nped - run_size / 2.0
    return q * d


def combined(tea_sd: float, se_sd: float, run_size: float, ped: float,
             nped: float | None) -> tuple[float, float]:
    """(E(Nu), E(Nuf)) when a shadow PBRTQC watches alongside the unchanged IQC."""
    d_pe = delta_pe(tea_sd, se_sd)
    d_iqc = distance_iqc(run_size, ped)
    nuf_iqc = e_nuf(tea_sd, se_sd, run_size, ped)
    if nped is None or not np.isfinite(nped) or nped >= d_iqc:
        return d_pe * d_iqc, nuf_iqc
    return d_pe * nped, min(nuf_iqc, d_pe * _final_distance_pbrtqc(nped, run_size, ped))


def risk_profile(tea_sd: float, run_size: float, se_grid, ped_grid,
                 nped_grid=None) -> dict:
    """E(Nu)/E(Nuf) across error sizes for IQC alone and, optionally, IQC + shadow PBRTQC."""
    se_grid = np.asarray(se_grid, float)
    ped_grid = np.asarray(ped_grid, float)
    nu = np.array([e_nu(tea_sd, s, run_size, p) for s, p in zip(se_grid, ped_grid)])
    nuf = np.array([e_nuf(tea_sd, s, run_size, p) for s, p in zip(se_grid, ped_grid)])
    out = dict(se=se_grid.tolist(), ped=ped_grid.tolist(), enu=nu.tolist(), enuf=nuf.tolist(),
               max_enuf=float(nuf.max()), se_at_max_enuf=float(se_grid[int(np.argmax(nuf))]),
               max_enu=float(nu.max()), run_size=float(run_size), tea_sd=float(tea_sd))
    if nped_grid is not None:
        cu, cf = zip(*[combined(tea_sd, s, run_size, p, n)
                       for s, p, n in zip(se_grid, ped_grid, nped_grid)])
        out.update(enu_combined=list(cu), enuf_combined=list(cf),
                   max_enuf_combined=float(np.max(cf)), max_enu_combined=float(np.max(cu)),
                   nped=list(nped_grid))
    return out


def run_size_for_target(tea_sd: float, target_max_enuf: float, se_grid, ped_grid,
                        lo=1.0, hi=100000.0) -> float:
    """Largest run size whose MaxE(Nuf) stays at or below the target (E(Nuf) is linear in N)."""
    def f(n):
        return max(e_nuf(tea_sd, s, n, p) for s, p in zip(se_grid, ped_grid))
    if f(lo) > target_max_enuf:
        return float("nan")
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if f(mid) > target_max_enuf:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.001:
            break
    return float(lo)


def effective_run_size(run_sizes) -> float:
    """Length-weighted (inspection-paradox) run size for E(Nuf).

    The error onset is uniform over patient results, so a long QC interval is more likely to
    contain it: P(onset in interval i) is proportional to N_i. The expected number of
    post-onset results remaining in that interval is therefore

        E[N^2] / (2 E[N])

    so pass N_eff = E[N^2]/E[N] as `n_onset` to e_nuf / e_nu, and the plain mean E[N] as
    `run_size` for the later runs (Parvin 2008: E(N_0) vs E(N_B)). N_eff exceeds the mean
    whenever the intervals vary (weekday vs weekend, extra QC runs); using the plain mean for
    the onset run underestimates patient risk, using N_eff for every run overestimates it.
    """
    n = np.asarray([x for x in run_sizes if np.isfinite(x) and x > 0], float)
    if n.size == 0:
        return float("nan")
    return float((n ** 2).sum() / n.sum())
