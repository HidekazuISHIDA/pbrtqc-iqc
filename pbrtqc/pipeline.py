"""Run the whole analysis, or part of it, in a fixed order - the same commands for the demo and the real data.

  python -m pbrtqc.pipeline --list                 show the steps
  python -m pbrtqc.pipeline                        run every step (profile from pbrtqc/config.py)
  python -m pbrtqc.pipeline --group select         run one group (prepare, select, fusion, checks, monitor)
  python -m pbrtqc.pipeline --from fusion_main     resume from a step

Each step is one script in scripts/ run in a fresh Python process. A run log (<work>/run_log.json) records the code
version, package versions, the start/end and exit status of every step and the SHA-256 of the raw input files, so a run
can be traced afterwards. Only aggregate result tables are written to <work>/tables.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata

from . import config as C

LOW5 = "Na,ALB,Cl,Ca,Mg"
#        id                  group      script                           args                                   description (ja)
STEPS = [
    ("synthetic",          "prepare", "00_make_synthetic_raw.py",      [],                                    "合成 raw データを作る（demo のみ）"),
    ("cache",              "prepare", "01_build_cache.py",             [],                                    "raw を読み、仮名化したキャッシュを作る"),
    ("flow",               "prepare", "51_methods_flow.py",            [],                                    "選択の流れと項目別の件数（Methods、Table S1）"),
    ("cohort",             "prepare", "59_cohort.py",                  [],                                    "患者背景（Methods）"),
    ("select",             "select",  "52_select_validate.py",         [],                                    "開発期間で設定を選び、検証年で評価（主解析）"),
    ("select_final",       "select",  "52_select_validate.py",         ["--value=final"],                     "同上、最終報告値（感度分析）"),
    ("select_noprune",     "select",  "52_select_validate.py",         ["--no-prune"],                        "3 成分を事前の絞り込みなしで総当たり（確認）"),
    ("prune_check",        "select",  "52c_prune_check.py",            [],                                    "絞り込みあり／なしの採用設定を比較"),
    ("fusion_main",        "fusion",  "53_iqc_fusion.py",              [],                                    "IQC + PBRTQC：検出時間と患者リスク（主解析）"),
    ("fusion_loo",         "fusion",  "53_iqc_fusion.py",              ["--loo"],                             "各成分を 1 つずつ外す"),
    ("sens_final",         "fusion",  "53_iqc_fusion.py",              ["--value=final"],                     "感度：最終報告値（設定を選び直す）"),
    ("sens_final_fixed",   "fusion",  "53_iqc_fusion.py",              ["--value=final", "--fixed-settings"], "感度：最終報告値（主解析の設定）"),
    ("sens_unit2",         "fusion",  "53_iqc_fusion.py",              ["--unit=2"],                          "感度：2 号機に主解析の設定"),
    ("sens_delay30",       "fusion",  "53_iqc_fusion.py",              ["--delay=30"],                        "感度：結果が到着 30 分後"),
    ("sens_delay60",       "fusion",  "53_iqc_fusion.py",              ["--delay=60"],                        "感度：結果が到着 60 分後"),
    ("sens_bias",          "fusion",  "53_iqc_fusion.py",              ["--bias-frac=0.25", f"--items={LOW5}"], "感度：バイアス 0.25 × TEa"),
    ("sens_cva",           "fusion",  "53_iqc_fusion.py",              ["--cva-mult=1.25", f"--items={LOW5}"],  "感度：CVa × 1.25"),
] + [
    (f"whatif_{cap}_{qc or 'none'}", "fusion", "53_iqc_fusion.py",
     [f"--cap={cap}", f"--items={LOW5}"] + ([f"--extra-qc={qc}"] if qc else []), f"試算：上限 {cap}、追加 QC {qc or 'なし'}")
    for cap in ("0.25", "0.5", "1.0") for qc in ("", "17", "17,1")
] + [
    ("shadow",             "checks",  "54_shadow_concordance.py",      [],                                    "実データのシャドー運用と QC 段差の突き合わせ"),
    ("limit_widths",       "checks",  "55_limit_widths.py",            [],                                    "管理限界の幅（Figure 2）"),
    ("parvin",             "checks",  "56_parvin_check.py",            [],                                    "Parvin 式との照合（実データ条件）"),
    ("parvin_stages",      "checks",  "56b_parvin_stages.py",          [],                                    "Parvin 式との段階的照合（合成）"),
    ("fig1_week",          "checks",  "57_fig1_week.py",               [],                                    "Alb の 1 週間（Figure 1）"),
    ("loo_summary",        "checks",  "58_loo_summary.py",             [],                                    "成分除去の要約"),
    ("paired_ci",          "checks",  "58b_paired_ci.py",              [],                                    "対応のある 95% 区間"),
    ("bias_sign",          "checks",  "58c_bias_by_sign.py",           [],                                    "バイアス感度の誤差の向き別"),
    ("monitor",            "monitor", "60_build_monitor.py",           [],                                    "モニター画面のデータ"),
]
GROUPS = ["prepare", "select", "fusion", "checks", "monitor"]


def steps(profile: str = C.PROFILE):
    return [s for s in STEPS if not (s[0] == "synthetic" and profile != "demo")]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def input_hashes() -> dict:
    files = sorted(C.LIS_RAW.glob("*.xlsx")) + sorted(C.LIS_RAW.glob("*.csv")) + [C.QC_RAW]
    return {p.name: _sha256(p) for p in files if p.exists()}


def environment() -> dict:
    pk = {}
    for name in ("numpy", "pandas", "scipy", "openpyxl"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=pk, code_version=_git_commit())


def run(selected=None, log_name="run_log.json"):
    """Run the given step ids (default: all, in order). Returns the run log."""
    todo = [s for s in steps() if selected is None or s[0] in selected]
    log_path = C.WORK / log_name
    log = dict(profile=C.PROFILE, started=dt.datetime.now().isoformat(timespec="seconds"), environment=environment(), steps=[])
    env = dict(os.environ, PBRTQC_PROFILE=C.PROFILE)
    for sid, group, script, args, desc in todo:
        print(f"\n=== {sid}: {desc}  [{script} {' '.join(args)}]", flush=True)
        t0 = time.time()
        # stream the step's output line by line (so it also appears in a Jupyter cell)
        with subprocess.Popen([sys.executable, "-u", str(C.ROOT / "scripts" / script), *args], cwd=C.ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            for line in p.stdout:
                print("   ", line, end="", flush=True)
        r = p
        rec = dict(step=sid, group=group, script=script, args=args, seconds=round(time.time() - t0, 1), returncode=r.returncode)
        log["steps"].append(rec)
        if sid == "cache" or (sid == "synthetic"):
            log["inputs_sha256"] = input_hashes()
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(f"--- {sid}: {'OK' if r.returncode == 0 else 'FAILED'} ({rec['seconds']} s)", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"step {sid} failed (exit {r.returncode}); later steps were not run")
    log["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--group", choices=GROUPS)
    ap.add_argument("--only", nargs="+", metavar="STEP")
    ap.add_argument("--from", dest="start", metavar="STEP")
    a = ap.parse_args(argv)
    ids = [s[0] for s in steps()]
    if a.list:
        print(C.describe())
        for sid, group, script, args, desc in steps():
            print(f"  {group:8s} {sid:20s} {desc}")
        return
    sel = None
    if a.group:
        sel = [s[0] for s in steps() if s[1] == a.group]
    if a.only:
        bad = set(a.only) - set(ids)
        if bad:
            raise SystemExit(f"unknown step(s): {sorted(bad)}")
        sel = a.only
    if a.start:
        sel = ids[ids.index(a.start):]
    print(C.describe())
    run(sel)


if __name__ == "__main__":
    main()
