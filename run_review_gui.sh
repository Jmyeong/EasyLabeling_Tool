#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
exec "${PYTHON}" "${ROOT}/review_gui.py" \
  --dataset-root "${DATA_ROOT:-${ROOT}/datasets}" \
  --host "${HOST_BIND:-127.0.0.1}" --port "${PORT:-8765}" "$@"
