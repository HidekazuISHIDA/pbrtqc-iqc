# PBRTQC + IQC: patient-based real-time quality control combined with internal QC

Code and a synthetic demonstration for the study *Combining patient-based real-time quality control with internal quality
control to reduce patient risk* (Clinica Chimica Acta, submitted).

**Live demo (synthetic data only):** https://hidekazuishida.github.io/pbrtqc-iqc/demo/ (GitHub Pages), or open `demo/index.html` through a local web server.
The demo replays January 2025 for one simulated analyzer with the 20 study analytes; a systematic error of +1.5 × TEa on Ca is
injected on 20–23 January. Useful URL parameters: `?t=2025-01-21T10:00&item=Ca`, `&play=1&speed=250&until=2025-01-24T00:00`,
`&lang=ja`, `&theme=light`.

## What the code does
| Module / script | Purpose |
|---|---|
| `pbrtqc/realsim.py` | Patient streams, truncated moving average (MA), average of patient deltas (AoD), even check, control limits (mean ± 3 SD of the development period), alarm episodes, error injection |
| `pbrtqc/engine.py` | Moving statistics and truncation |
| `pbrtqc/iqc.py`, `pbrtqc/iqc_model.py` | Westgard multirule, QC events, robust Total CV |
| `pbrtqc/risk.py` | Parvin's expected number of unreliable final results, E(Nuf) |
| `pbrtqc/lis.py`, `pbrtqc/iqc_data.py` | Loaders for the laboratory-information-system exports used in the study (format described in the docstrings) |
| `scripts/52_select_validate.py` | Select settings in the development period under an alarm limit; evaluate in the validation year |
| `scripts/53_iqc_fusion.py` | IQC + PBRTQC fusion simulation, detection time and MaxE(Nuf) |
| `scripts/54_shadow_concordance.py` | Real-data shadow operation and concordance with QC shifts |
| `scripts/55`–`57` | Control-limit widths, check against Parvin's formula, Figure 1 data |
| `scripts/60_build_monitor.py` | Data for the monitoring screen |
| `scripts/demo_make_data.py` | Synthetic data for the demo (no patient data) |
| `monitor/index.html` | Monitoring screen (single file, no external libraries) |

## Data availability
The patient and QC data contain confidential clinical information and are **not** included. The scripts expect exports
in the format described in `pbrtqc/lis.py` and `pbrtqc/iqc_data.py`; paths are relative to the repository root
(`data/raw/...`, `data/qc/...`). All analyses run on a local computer; no data leave the institution.

## Requirements
Python ≥ 3.11, see `requirements.txt`.

## AI assistance
Analysis scripts and the monitoring-screen prototype were written with the assistance of an AI coding assistant
(Claude, Anthropic). All code was reviewed by the authors and the results were verified independently.

## Licence
MIT (see `LICENSE`).
