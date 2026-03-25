#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ ! -d ".venv" ]; then
  echo "Fehlende .venv. Bitte zuerst ./scripts/setup.sh ausfuehren." >&2
  exit 1
fi

git -C PointNeXt submodule update --init --recursive

./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1 torchvision==0.20.1

./.venv/bin/pip install \
  "numpy<2" \
  easydict \
  multimethod \
  termcolor \
  shortuuid \
  tensorboard \
  gdown \
  ninja \
  protobuf==3.19.4

echo
echo "PointNeXt CPU-Inferenzumgebung ist in .venv installiert."
echo "Falls du Klassifikation aktivieren willst, lege noch den Checkpoint unter ckpt/ ab."
