#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

UNIT_NAME="${UNIT_NAME:-tracking-pipeline-live.service}"
ENV_FILE="${ENV_FILE:-/etc/default/tracking-pipeline-live}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-$(id -un)}}"

UNIT_TEMPLATE="${REPO_ROOT}/deploy/systemd/tracking-pipeline-live.service.tpl"
ENV_TEMPLATE="${REPO_ROOT}/deploy/systemd/tracking-pipeline-live.env.example"
UNIT_TARGET="${SYSTEMD_DIR}/${UNIT_NAME}"

if [[ ! -f "${UNIT_TEMPLATE}" ]]; then
    echo "Unit template not found: ${UNIT_TEMPLATE}" >&2
    exit 66
fi

if [[ ! -f "${ENV_TEMPLATE}" ]]; then
    echo "Environment template not found: ${ENV_TEMPLATE}" >&2
    exit 66
fi

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run this script with sudo so it can write to ${SYSTEMD_DIR}." >&2
    exit 77
fi

TMP_UNIT="$(mktemp)"
trap 'rm -f "${TMP_UNIT}"' EXIT

sed \
    -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
    -e "s|@SERVICE_USER@|${SERVICE_USER}|g" \
    -e "s|@ENV_FILE@|${ENV_FILE}|g" \
    "${UNIT_TEMPLATE}" > "${TMP_UNIT}"

install -D -m 0644 "${TMP_UNIT}" "${UNIT_TARGET}"

if [[ ! -f "${ENV_FILE}" ]]; then
    install -D -m 0644 "${ENV_TEMPLATE}" "${ENV_FILE}"
    echo "Created ${ENV_FILE} from template."
else
    echo "Keeping existing environment file: ${ENV_FILE}"
fi

systemctl daemon-reload

cat <<EOF
Installed ${UNIT_TARGET}

Next steps:
  1. Edit ${ENV_FILE} and set TRACKING_PIPELINE_CONFIG to your real live config.
  2. Enable and start the service:
     sudo systemctl enable --now ${UNIT_NAME}
  3. Follow the logs:
     journalctl -fu ${UNIT_NAME}

Useful commands:
  sudo systemctl restart ${UNIT_NAME}
  sudo systemctl status ${UNIT_NAME}
  sudo systemctl stop ${UNIT_NAME}
EOF
