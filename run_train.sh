#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${ROOT}/.ultralytics_yolo26}"

"${PYTHON}" "${ROOT}/train_yolo26_mtl.py" \
  --model "${MODEL:-${ROOT}/checkpoints/yolo26s.pt}" \
  --dataset-root "${DATA_ROOT:-${ROOT}/datasets}" \
  --output-dir "${OUTPUT_DIR:-${ROOT}/runs/yolo26s_rgb_nir_gated_fusion_dated_mtl}" \
  --input-mode "${INPUT_MODE:-gated}" \
  --fusion-repo "${FUSION_REPO:-external/Pixel_aligned_RGB_NIR_Stereo}" \
  --fusion-checkpoint "${FUSION_CHECKPOINT:-external/Pixel_aligned_RGB_NIR_Stereo/weights/model_image_fusion.pth}" \
  --epochs "${EPOCHS:-100}" \
  --batch "${BATCH:-8}" \
  --device "${DEVICE:-0}" \
  --workers "${WORKERS:-8}" \
  --height "${HEIGHT:-352}" \
  --width "${WIDTH:-640}" \
  --depth-weight "${DEPTH_WEIGHT:-0.5}" \
  --head-warmup-epochs "${HEAD_WARMUP_EPOCHS:-5}" \
  "$@"
