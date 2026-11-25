# PBRTQC + IQC: patient-based real-time quality control combined with internal QC

Code, a synthetic data set and a reproducible pipeline for the study of patient-based real-time quality control (PBRTQC)
designed for the daily case mix of a hospital laboratory and combined with internal quality control (IQC)
(Clinica Chimica Acta, in preparation).

**Live demo of the monitoring screen (synthetic data only):** https://hidekazuishida.github.io/pbrtqc-iqc/demo/
(useful URL parameters: `?t=2025-01-21T10:00&item=Ca`, `&play=1&speed=250&until=2025-01-24T00:00`, `&lang=ja`, `&theme=light`).

## How the code was built and how it is verified
| Stage | Who | What | Data |
|---|---|---|---|
| 1. Build | AI coding assistant (Claude, Anthropic), reviewed by the authors | Analysis methods, monitoring screen and verification method, run end to end on synthetic data | Synthetic exports (`scripts/00_make_synthetic_raw.py`) |
| 2. Environment | AI coding assistant | VS Code + Jupyter environment and instructions to run the same code on real data | none |
| 3. Reproduction | An author | Runs the published code on the laboratory's own exports on a local computer | Real data (never leave the computer) |
| 4. Verification | An author | Compares every result table cell by cell with the tables of the reported analysis (`notebooks/06_verify.ipynb`) | Aggregate tables only |

There is one code path; the demo and the real analysis differ only in which exports are read (`pbrtqc/config.py`).
Step-by-step instructions (Japanese): [`docs/REPRODUCE_ja.md`](docs/REPRODUCE_ja.md).

## Quick start (demo, macOS)
```bash
./setup_mac.sh                               # .venv with pinned packages + commit guard
.venv/bin/python -m pbrtqc.pipeline          # synthetic exports -> all analyses (about 10 min)
.venv/bin/python -m pbrtqc.verify            # compare with reference/demo_tables (all tables should agree)
```
Or open the folder in VS Code and run `notebooks/00_setup.ipynb` … `06_verify.ipynb` in order (kernel: `.venv`).

## Layout
| Path | Purpose |
|---|---|
| `pbrtqc/config.py` | Profile `demo` (synthetic, `work_demo/`) or `real` (paths in `pbrtqc.local.json`, never committed; results in `work/`) |
| `pbrtqc/pipeline.py` | Runs all steps in a fixed order and writes a run log (code version, package versions, timings, SHA-256 of inputs) |
| `pbrtqc/verify.py` | Cell-by-cell comparison of 43 result tables with reference tables (relative tolerance 1e-9) |
| `pbrtqc/lis.py`, `pbrtqc/iqc_data.py` | Loaders for the LIS and IQC exports (.xlsx, or .csv with the same columns); IDs replaced by sequential keys |
| `pbrtqc/realsim.py`, `engine.py` | Patient streams, truncated moving average, average of patient deltas (AoD), even check, control limits, alarm episodes |
| `pbrtqc/iqc.py`, `iqc_model.py`, `risk.py` | Westgard multirule, QC events, robust CV, Parvin's E(Nuf) |
| `scripts/00`–`01`, `51`, `59` | Synthetic exports, caches, selection flow, cohort description |
| `scripts/52`–`52c` | Setting selection in 2023–2024 under an alarm limit, validation in 2025, exhaustive-search check |
| `scripts/53` | IQC + PBRTQC simulation: detection time, MaxE(Nuf), component removal, sensitivity and what-if analyses |
| `scripts/54`–`58c` | Shadow operation, limit widths, checks against Parvin's formula, Figure 1 data, paired intervals, bias by sign |
| `scripts/60`, `monitor/index.html` | Monitoring screen (single file, no external libraries) |
| `notebooks/` | The same steps as Jupyter notebooks with Japanese explanations |
| `reference/demo_tables/` | Result tables of the synthetic data (reference for the demo verification) |
| `tools/guard.py`, `.githooks/` | Refuses commits that could carry patient data |

## Data availability
The patient and QC data contain confidential clinical information and are **not** included. The loaders document the
expected export format; the synthetic generator writes files in the same layout.

## Requirements
Python 3.14 and the pinned versions in `requirements.txt` (identical versions give identical numbers).

## Licence
MIT (see `LICENSE`).
