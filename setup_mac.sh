#!/bin/zsh
# One-time setup on macOS: a private Python environment (.venv) with the pinned packages, and the commit guard.
set -e
cd "$(dirname "$0")"
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
echo "準備完了: VS Code で notebooks/00_setup.ipynb を開き、カーネルに .venv を選んでください。"
