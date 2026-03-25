#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Kein passender Python-Interpreter gefunden. Bitte Python 3.10 oder neuer installieren." >&2
  exit 1
fi

echo "Using ${PYTHON_BIN}"
"${PYTHON_BIN}" -m venv .venv

source .venv/bin/activate
python --version
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

echo
echo "Setup abgeschlossen."
echo "Aktivieren mit: source .venv/bin/activate"
echo "CLI testen mit: tracking-pipeline --help"
