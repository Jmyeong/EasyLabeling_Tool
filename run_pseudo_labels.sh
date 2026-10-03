#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${ROOT}/.ultralytics_yolo26}"
"${PYTHON}" "${ROOT}/generate_pseudo_labels.py" \
  --dataset-root "${DATA_ROOT:-${ROOT}/datasets}" \
  --checkpoint "${CHECKPOINT:-${ROOT}/checkpoints/best_joint.pt}" \
  --device "${DEVICE:-0}" --batch "${BATCH:-8}" --workers "${WORKERS:-4}" "$@"
