#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BIN="${TRACKING_PIPELINE_BIN:-${REPO_ROOT}/.venv/bin/tracking-pipeline}"
CONFIG="${TRACKING_PIPELINE_CONFIG:-}"

if [[ -z "${CONFIG}" ]]; then
    echo "TRACKING_PIPELINE_CONFIG is not set." >&2
    exit 64
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Tracking pipeline binary not found or not executable: ${BIN}" >&2
    exit 66
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Tracking pipeline config not found: ${CONFIG}" >&2
    exit 66
fi

cd "${REPO_ROOT}"
exec "${BIN}" run -c "${CONFIG}"
