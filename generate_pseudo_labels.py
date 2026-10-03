#!/usr/bin/env python3
"""Generate reviewable four-class detection and binary Road pseudo annotations.

Uses the existing joint RGB/NIR checkpoint and exactly its training letterbox.
No training or GUI is started. Original images, labels and manifests are untouched.
"""

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from dated_dataset import annotation_paths, read_manifest, resolve_file
from box_filter import filter_duplicate_boxes
from road_filter import keep_largest_road_blobs
from pseudo_index import rebuild_index
from prepare_raw_dy_from_instseg import CLASS_NAMES, yolo_line
from train import letterbox_all
from train_yolo26_mtl import select_device
from test_yolo26_mtl import load_model, decode_yolo26_detections, validate_test_environment

PROJECT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, help="Default: <dataset-root>/pseudo_labels")
    parser.add_argument("--dates", nargs="+", default=["260916"])
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--checkpoint", default=str(PROJECT / "checkpoints/best_joint.pt"))
    parser.add_argument("--model-template", default=str(PROJECT / "checkpoints/yolo26s.pt"))
    parser.add_argument("--input-mode", default="auto", choices=("auto", "gated", "scalar", "rgbn", "depth_nir", "depth_nir_gated", "hsvnet"))
    parser.add_argument("--fusion-repo", default="")
    parser.add_argument("--fusion-checkpoint", default="")
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--write-workers", type=int, default=4, help="Parallel PNG/preview writers")
    parser.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--box-dedup", action=argparse.BooleanOptionalAction, default=True, help="Same-class near-duplicate suppression")
    parser.add_argument("--box-dedup-iou", type=float, default=0.9, help="Suppress same-class boxes with IoU >= threshold")
    parser.add_argument("--road-threshold", type=float, default=0.50)
    parser.add_argument("--road-ignore-margin", type=float, default=0.10, help="Mark pixels within this margin of threshold as 255 (ignore)")
    parser.add_argument("--road-max-blobs", type=int, default=2, help="Keep at most N largest 8-connected road blobs; 0 disables filtering")
    parser.add_argument("--previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: select first N frames; 0 means all")
    args = parser.parse_args()
    args.imgsz, args.min_depth, args.max_depth = None, 1.0, 35.0
    if not 0 < args.box_dedup_iou <= 1:
        parser.error("box-dedup-iou must be in (0,1]")
    if args.road_max_blobs < 0:
        parser.error("road-max-blobs must be nonnegative")
    if not 0 <= args.conf <= 1 or not 0 < args.road_threshold < 1:
        parser.error("Invalid confidence/road threshold")
    if not 0 <= args.road_ignore_margin < min(args.road_threshold, 1-args.road_threshold):
        parser.error("road-ignore-margin must fit inside [0,1] around threshold")
    if args.batch < 1 or args.workers < 0 or args.limit < 0 or args.max_det < 1 or args.write_workers < 1:
        parser.error("batch/max-det must be positive; workers/limit must be nonnegative")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class InferenceDataset(Dataset):
    def __init__(self, root, rows, size):
        self.root, self.rows, self.size = root, rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(resolve_file(self.root, row["image_file"])).convert("RGB"))
        nir = np.asarray(Image.open(resolve_file(self.root, row["nir_file"])).convert("L"))
        h, w = image.shape[:2]
        rgb, nir, _, _, _ = letterbox_all(
            image, nir, np.zeros((h, w), np.float32), np.zeros((h, w), np.uint8), [], self.size,
        )
        return {
            "index": index, "height": h, "width": w,
            "img": torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
            "nir": torch.from_numpy(nir).unsqueeze(0).float() / 255,
        }


def restore_predictions(detections, probability, height, width, size):
    """Undo padding before resize; boxes use the same scalar as letterbox_all."""
    th, tw = size
    scale = min(th / height, tw / width)
    nw, nh = round(width * scale), round(height * scale)
    px, py = (tw - nw) // 2, (th - nh) // 2
    probability = cv2.resize(probability[py:py+nh, px:px+nw], (width, height), interpolation=cv2.INTER_LINEAR)
    boxes = []
    for cls, x1, y1, x2, y2, score in detections:
        if not 0 <= cls < len(CLASS_NAMES) or not np.isfinite([x1, y1, x2, y2, score]).all():
            raise ValueError(f"Invalid detector output: {detections}")
        xyxy = [float(np.clip((x1-px)/scale, 0, width)), float(np.clip((y1-py)/scale, 0, height)),
                float(np.clip((x2-px)/scale, 0, width)), float(np.clip((y2-py)/scale, 0, height))]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        boxes.append({"class_id": cls, "class_name": CLASS_NAMES[cls], "bbox_xyxy": xyxy, "confidence": score})
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite road probability")
    return boxes, probability


def save_preview(path, image, road, boxes):
    canvas = image.copy()
    for value, color in [(1, (30, 210, 60)), (255, (255, 180, 30))]:
        selected = road == value
        canvas[selected] = (0.55*canvas[selected] + 0.45*np.array(color)).astype(np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = map(round, box["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 80, 80), 2)
        text = f"{box['class_name']} {box['confidence']:.2f}"
        cv2.putText(canvas, text, (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(canvas, "PSEUDO / PENDING | green: road | orange: uncertain", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    Image.fromarray(canvas).save(path, quality=88)


def write_sample(args, root, row, boxes, probability, signature):
    raw_boxes = boxes
    removed = []
    if args.box_dedup:
        boxes, removed = filter_duplicate_boxes(boxes, args.box_dedup_iou)
    paths = annotation_paths(row)
    if not args.previews:
        paths["preview_file"] = ""
    for value in paths.values():
        if value:
            (root / value).parent.mkdir(parents=True, exist_ok=True)
    h, w = probability.shape
    road = (probability >= args.road_threshold).astype(np.uint8)
    uncertain = np.abs(probability - args.road_threshold) < args.road_ignore_margin
    road[uncertain] = 255
    road_filter_info = None
    if args.road_max_blobs:
        road, road_filter_info = keep_largest_road_blobs(road, args.road_max_blobs)
    Image.fromarray(road).save(root / paths["road_file"], compress_level=1)
    Image.fromarray(np.rint(probability * 65535).astype(np.uint16)).save(root / paths["road_probability_file"], compress_level=1)
    lines = [yolo_line(box, w, h) for box in boxes]
    (root / paths["detection_file"]).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if args.previews:
        image = np.asarray(Image.open(resolve_file(args.dataset_root, row["image_file"])).convert("RGB"))
        save_preview(root / paths["preview_file"], image, road, boxes)
    record = {
        "schema_version": 1, "label_source": "pseudo", "review_status": "pending",
        "image_file": row["image_file"], "nir_file": row["nir_file"],
        "image_sha256": row.get("image_sha256", ""),
        "date": row["date"], "split": row["split"], "sequence": row["sequence"],
        "width": w, "height": h, **paths,
        "generation_signature": signature,
        "detections": boxes, "detection_count": len(boxes),
        "road_fraction": float((road == 1).mean()), "uncertain_fraction": float(uncertain.mean()),
        "review_reasons": (["no_detections"] if not boxes else []) +
                          (["uncertain_road"] if uncertain.mean() > 0.10 else []) +
                          (["low_confidence_detection"] if any(b["confidence"] < 0.4 for b in boxes) else []),
    }
    if args.box_dedup:
        record["box_dedup"] = {"method":"same_class_nms", "iou_threshold":args.box_dedup_iou,
                               "before":len(raw_boxes), "after":len(boxes), "removed":removed}
        if removed:
            record["raw_detections"] = raw_boxes
    if road_filter_info is not None:
        record["road_blob_filter"] = road_filter_info
    # Written last: this record is the completion marker used when resuming.
    atomic_json(root / paths["annotation_file"], record)
    return record


def main():
    args = parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics_yolo26"))
    args.dataset_root = args.dataset_root.expanduser().resolve()
    root = (args.output_root or args.dataset_root / "pseudo_labels").expanduser().resolve()
    if root == args.dataset_root or (root / "manifest.csv").exists() and not (root / "run.json").exists():
        raise ValueError("Use a separate pseudo-label output directory")
    rows = [r for r in read_manifest(args.dataset_root)
            if r["date"] in args.dates and r["split"] in args.splits
            and not r["semseg_file"] and not r["instseg_file"]]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No unlabeled frames match dates/splits")
    seen = set()
    for row in rows:
        key = annotation_paths(row)["annotation_file"]
        if key in seen:
            raise ValueError(f"Duplicate output identity: {key}")
        seen.add(key)
        for field in ("image_file", "nir_file"):
            if not row[field] or not Path(resolve_file(args.dataset_root, row[field])).is_file():
                raise FileNotFoundError(f"Missing {field}: {row['image_file']}")
    validate_test_environment(args)
    config = {
        "schema_version": 1, "dataset_root": str(args.dataset_root),
        "dataset_manifest_sha256": sha256(args.dataset_root / "manifest.csv"),
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "model_template_sha256": sha256(args.model_template),
        "input_size": [args.height, args.width], "confidence": args.conf, "max_det": args.max_det,
        "road_threshold": args.road_threshold, "road_ignore_margin": args.road_ignore_margin,
        "previews": args.previews, "class_names": CLASS_NAMES,
    }
    if args.box_dedup:
        config["box_dedup"] = {"method":"same_class_nms", "iou_threshold":args.box_dedup_iou}
    if args.road_max_blobs:
        config["road_blob_filter"] = {"method":"largest_road_components", "connectivity":8,
                                    "max_blobs":args.road_max_blobs}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    # Prevent two writers from sharing a directory; released even on exceptions.
    import fcntl
    with (root / ".generation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = root / "run.json"
        if run_path.exists():
            previous = json.loads(run_path.read_text(encoding="utf-8"))
            if previous["generation_signature"] != signature:
                raise ValueError("Output contains a different generation configuration; use a new --output-root")
        else:
            atomic_json(run_path, {**config, "generation_signature": signature, "created_utc": datetime.now(timezone.utc).isoformat()})
        pending = []
        for row in rows:
            record_path = root / annotation_paths(row)["annotation_file"]
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if record["generation_signature"] != signature:
                    raise ValueError(f"Generation mismatch: {record_path}")
                for key in ("detection_file", "road_file", "road_probability_file", "preview_file"):
                    if record[key] and not (root / record[key]).is_file():
                        raise FileNotFoundError(f"Incomplete existing annotation (preserved): {record_path}, {key}")
            else:
                pending.append(row)
        print(f"Selected={len(rows)} existing={len(rows)-len(pending)} pending={len(pending)} output={root}", flush=True)
        if pending:
            device = select_device(args.device)
            model, checkpoint = load_model(args, device)
            atomic_json(root / "model.json", {"epoch": checkpoint.get("epoch"), "architecture": checkpoint.get("architecture"), "input_mode": args.input_mode})
            del checkpoint
            dataset = InferenceDataset(args.dataset_root, pending, (args.height, args.width))
            loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False, pin_memory=device.type == "cuda")
            processed = 0
            try:
                with torch.inference_mode(), ThreadPoolExecutor(max_workers=args.write_workers) as writers:
                    for batch in loader:
                        detection, _, road = model(batch["img"].to(device), batch["nir"].to(device))
                        detections = decode_yolo26_detections(detection, args)
                        probabilities = road.sigmoid().cpu().numpy()
                        writes = []
                        for i, det in enumerate(detections):
                            row = pending[int(batch["index"][i])]
                            boxes, probability = restore_predictions(det, probabilities[i], int(batch["height"][i]), int(batch["width"][i]), (args.height, args.width))
                            writes.append(writers.submit(write_sample, args, root, row, boxes, probability, signature))
                        for write in writes:
                            write.result()
                            processed += 1
                        if processed % (args.batch*10) == 0 or processed == len(pending):
                            print(f"Generated {processed}/{len(pending)}", flush=True)
            finally:
                rebuild_index(root)
        summary = rebuild_index(root)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
