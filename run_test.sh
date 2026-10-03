#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${ROOT}/.ultralytics_yolo26}"

EXTRA_ARGS=()
if [[ "${SAVE_PER_SAMPLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save-per-sample)
fi

"${PYTHON}" "${ROOT}/test_yolo26_mtl.py" \
  --checkpoint "${CHECKPOINT:-${ROOT}/runs/yolo26s_rgb_nir_gated_fusion_mtl/best_depth.pt}" \
  --model-template "${MODEL:-${ROOT}/checkpoints/yolo26s.pt}" \
  --dataset-root "${DATA_ROOT:-${ROOT}/datasets}" \
  --output-root "${OUTPUT_ROOT:-${ROOT}/results/yolo26s_rgb_nir_gated_fusion_mtl_best_depth}" \
  --device "${DEVICE:-0}" \
  --batch "${BATCH:-1}" \
  --workers "${WORKERS:-8}" \
  --height "${HEIGHT:-352}" \
  --width "${WIDTH:-640}" \
  --eval_depth_in "${EVAL_DEPTH_IN:-15.0}" \
  --nir-ablation "${NIR_ABLATION:-normal}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
