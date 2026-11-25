"""Commit guard: refuse commits that could carry patient data (run by .githooks/pre-commit).

Blocks: anything under data/, work/, work_demo/ or outputs/; pbrtqc.local.json; *.pkl, *.xlsx; .csv/.npz outside
reference/demo_tables; notebooks that still contain outputs; text files containing long digit strings (IDs) or the name
of the patient-ID column of the export. `python tools/guard.py --strip notebooks/*.ipynb` clears notebook outputs.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

BLOCK_DIRS = ("data/", "work/", "work_demo/", "outputs/")
ID_PATTERN = re.compile(r"(?<![\d.])\d{9,}(?![\d.])|PID20\d{6}")


def strip(paths):
    for p in paths:
        nb = json.loads(Path(p).read_text(encoding="utf-8"))
        for c in nb.get("cells", []):
            if c.get("cell_type") == "code":
                c["outputs"], c["execution_count"] = [], None
        Path(p).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("cleared outputs:", p)


def check(staged):
    problems = []
    for f in staged:
        p = Path(f)
        if not p.exists():
            continue
        if f.startswith(BLOCK_DIRS) or f == "pbrtqc.local.json" or p.suffix in (".pkl", ".xlsx"):
            problems.append(f"{f}: data, cache or local settings must not be committed")
            continue
        if p.suffix in (".csv", ".npz") and not f.startswith("reference/demo_tables/"):
            problems.append(f"{f}: result tables are committed only under reference/demo_tables")
            continue
        if p.suffix == ".ipynb":
            nb = json.loads(p.read_text(encoding="utf-8"))
            if any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code"):
                problems.append(f"{f}: notebook has outputs - run: .venv/bin/python tools/guard.py --strip {f}")
            continue
        if p.suffix in (".py", ".md", ".json", ".txt", ".html", ".sh", ".csv") and p.stat().st_size < 5_000_000:
            m = ID_PATTERN.search(p.read_text(encoding="utf-8", errors="ignore"))
            if m and not f.startswith("reference/demo_tables/"):
                problems.append(f"{f}: contains '{m.group(0)[:12]}…' (looks like an ID)")
    return problems


if __name__ == "__main__":
    if sys.argv[1:2] == ["--strip"]:
        strip(sys.argv[2:])
        sys.exit(0)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], capture_output=True, text=True).stdout.split()
    probs = check(staged)
    if probs:
        print("コミットを止めました（患者データの混入防止）:\n  " + "\n  ".join(probs))
        sys.exit(1)
