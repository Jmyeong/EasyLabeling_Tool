#!/usr/bin/env python3
"""Evaluate and visualize YOLO Detection + Depth + Road segmentation.

This test script uses the model and dataset definitions from ``train.py`` so
training and evaluation always share the same preprocessing and architecture.
One model forward produces all three task outputs.
"""

import argparse
import csv
import html
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace
try:
    # Ultralytics >= 8.4 (YOLO26)
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    # Ultralytics 8.3 (existing YOLO11 environment)
    from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.torch_utils import get_flops

from train import (
    ThreeTaskDataset,
    YoloDepthDetRoad,
    collate,
    normalize_image_size,
    resolve_image_size,
)
from visualize_yolo_test import (
    CLASS_NAMES,
    draw_boxes,
    load_font,
    match_detections,
)


# -----------------------------------------------------------------------------
# Arguments and model loading
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test joint YOLO detection, metric depth, and Road segmentation"
    )
    parser.add_argument(
        "--checkpoint",
        default="./runs/yolo11s_mtl/best_joint.pt",
        help="3-task checkpoint created by train.py",
    )
    parser.add_argument(
        "--detector-init",
        default="",
        help="YOLO detector checkpoint. Empty uses checkpoint args['model'].",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
    )
    parser.add_argument(
        "--output-root",
        default="./test_depth_det_road_mtl",
    )
    parser.add_argument("--device", default="3")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=352, help="Network input height")
    parser.add_argument("--width", type=int, default=640, help="Network input width")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Backward-compatible alias for --width (height remains --height)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of unmeasured forwards before inference timing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N samples (0: full test set)",
    )

    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--eval-iou", type=float, default=0.30)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=35.0)
    parser.add_argument(
        "--eval-depth-in",
        "--eval_depth_in",
        type=float,
        default=None,
        help=(
            "Evaluate detection and depth only at GT depth <= this distance in meters "
            "(default: full configured depth range)"
        ),
    )
    parser.add_argument("--road-threshold", type=float, default=0.50)

    parser.add_argument("--sheet-count", type=int, default=36)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--no-images", action="store_true", help="Evaluate without saving images")
    return parser.parse_args()


def select_device(device_text):
    if device_text == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or expose a GPU")
    return torch.device(f"cuda:{str(device_text).split(',')[0]}")


def synchronize(device):
    """Wait for queued CUDA kernels before reading the wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warm_up_model(model, device, image_size, runs):
    """Run unmeasured forwards to remove one-time initialization overhead."""
    if runs <= 0:
        return
    height, width = normalize_image_size(image_size)
    sample = torch.zeros((1, 3, height, width), device=device)
    with torch.inference_mode():
        for _ in range(runs):
            model(sample)
    synchronize(device)


def infer_detection_class_count(state_dict):
    """Read the number of YOLO classes from the final classification bias."""
    suffix = "detector.model.23.cv3.0.2.bias"
    for name, value in state_dict.items():
        if name.endswith(suffix):
            return int(value.numel())
    raise ValueError("Could not infer the detection class count from the checkpoint")


def resolve_detector_template(cli_path, checkpoint_args):
    """Find a YOLO checkpoint used only to reconstruct the detector graph."""
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Detector checkpoint not found: {path}")
        return path

    stored_path = Path(checkpoint_args.get("model", "")).expanduser()
    if stored_path.is_file():
        return stored_path.resolve()

    fallback = Path(__file__).resolve().with_name("yolo11s.pt")
    if fallback.exists():
        print(f"detector template fallback={fallback}")
        return fallback

    raise FileNotFoundError(
        "The detector path saved in the checkpoint is unavailable. "
        "Pass --detector-init with the matching YOLO base model."
    )


def load_model(args, device):
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"3-task checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    state_dict = checkpoint["model"]
    detector_path = resolve_detector_template(args.detector_init, checkpoint_args)
    detector = YOLO(detector_path).model
    detector_args = getattr(detector, "args", {})
    class_count = infer_detection_class_count(state_dict)
    if detector.model[-1].nc != class_count:
        detector = DetectionModel(
            detector.yaml,
            ch=detector.yaml.get("channels", 3),
            nc=class_count,
            verbose=False,
        )
    if not isinstance(detector_args, dict):
        detector_args = vars(detector_args)
    detector.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **detector_args})

    min_depth = float(checkpoint_args.get("min_depth", args.min_depth))
    model = YoloDepthDetRoad(detector, min_depth=min_depth)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), checkpoint


# -----------------------------------------------------------------------------
# Detection decoding and metrics
# -----------------------------------------------------------------------------


def decode_detections(raw_output, args):
    prediction = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    nms_results = non_max_suppression(
        prediction,
        conf_thres=args.conf,
        iou_thres=args.nms_iou,
        max_det=args.max_det,
        nc=len(CLASS_NAMES),
    )
    return [
        [
            (
                int(row[5].item()),
                *row[:4].detach().cpu().tolist(),
                float(row[4].item()),
            )
            for row in sample
        ]
        for sample in nms_results
    ]


def get_gt_boxes(batch, sample_index, image_size):
    image_height, image_width = normalize_image_size(image_size)
    selected = batch["batch_idx"] == sample_index
    classes = batch["cls"][selected, 0].detach().cpu().tolist()
    normalized_boxes = batch["bboxes"][selected].detach().cpu().tolist()

    boxes = []
    for class_id, (cx, cy, width, height) in zip(classes, normalized_boxes):
        boxes.append(
            (
                int(class_id),
                (cx - width / 2) * image_width,
                (cy - height / 2) * image_height,
                (cx + width / 2) * image_width,
                (cy + height / 2) * image_height,
                None,
            )
        )
    return boxes


def box_gt_depth(box, depth_gt, valid_depth, center_fraction=0.50):
    """Estimate a box distance using median GT depth in its central region."""
    height, width = depth_gt.shape[-2:]
    x1, y1, x2, y2 = (float(value) for value in box[1:5])
    box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    margin_x = box_width * (1.0 - center_fraction) / 2.0
    margin_y = box_height * (1.0 - center_fraction) / 2.0
    left = max(0, min(width, int(math.floor(x1 + margin_x))))
    right = max(0, min(width, int(math.ceil(x2 - margin_x))))
    top = max(0, min(height, int(math.floor(y1 + margin_y))))
    bottom = max(0, min(height, int(math.ceil(y2 - margin_y))))
    if right <= left or bottom <= top:
        return None

    region_valid = valid_depth[top:bottom, left:right]
    if not bool(region_valid.any()):
        return None
    values = depth_gt[top:bottom, left:right][region_valid]
    values = values[torch.isfinite(values)]
    return float(values.median().item()) if values.numel() else None


def filter_boxes_by_gt_depth(boxes, depth_gt, valid_depth, max_depth):
    """Keep boxes whose robust GT-depth estimate is within ``max_depth``."""
    if max_depth is None:
        return boxes
    return [
        box
        for box in boxes
        if (
            (distance := box_gt_depth(box, depth_gt, valid_depth)) is not None
            and distance <= max_depth
        )
    ]


def depth_evaluation_mask(target, valid, max_depth):
    """Build the valid-pixel mask for full-range or distance-limited depth metrics."""
    if max_depth is None:
        return valid
    return valid & torch.isfinite(target) & (target <= max_depth)


def update_detection_counts(accumulator, scope, image_counts):
    for class_id, counts in image_counts.items():
        for key in ("tp", "fp", "fn"):
            accumulator[(scope, class_id)][key] += counts[key]


def safe_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def finish_detection_metrics(accumulator, dates, args):
    rows = []
    for scope in ["all", *sorted(dates)]:
        class_totals = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            counts = dict(accumulator[(scope, class_id)])
            precision, recall, f1 = safe_prf(counts["tp"], counts["fp"], counts["fn"])
            class_totals.append(counts)
            rows.append(
                {
                    "scope": scope,
                    "class": class_name,
                    **counts,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confidence": args.conf,
                    "eval_iou": args.eval_iou,
                    "eval_depth_in": args.eval_depth_in,
                }
            )

        total = {
            key: sum(counts[key] for counts in class_totals)
            for key in ("tp", "fp", "fn")
        }
        precision, recall, f1 = safe_prf(total["tp"], total["fp"], total["fn"])
        rows.insert(
            len(rows) - len(CLASS_NAMES),
            {
                "scope": scope,
                "class": "All(micro)",
                **total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confidence": args.conf,
                "eval_iou": args.eval_iou,
                "eval_depth_in": args.eval_depth_in,
            },
        )
    return rows


# -----------------------------------------------------------------------------
# Depth and Road segmentation metrics
# -----------------------------------------------------------------------------


def update_depth_counts(accumulator, scope, prediction, target, valid):
    prediction = prediction[valid].clamp_min(1e-4)
    target = target[valid].clamp_min(1e-4)
    if prediction.numel() == 0:
        return

    difference = prediction - target
    ratio = torch.maximum(prediction / target, target / prediction)
    accumulator[scope]["valid_pixels"] += int(prediction.numel())
    accumulator[scope]["abs_rel_sum"] += float((difference.abs() / target).sum().item())
    accumulator[scope]["squared_sum"] += float(difference.square().sum().item())
    accumulator[scope]["delta1_sum"] += int((ratio < 1.25).sum().item())


def finish_depth_metrics(accumulator, dates, eval_depth_in=None):
    rows = []
    for scope in ["all", *sorted(dates)]:
        counts = accumulator[scope]
        pixels = int(counts["valid_pixels"])
        rows.append(
            {
                "scope": scope,
                "valid_pixels": pixels,
                "abs_rel": counts["abs_rel_sum"] / pixels if pixels else 0.0,
                "rmse": (counts["squared_sum"] / pixels) ** 0.5 if pixels else 0.0,
                "delta1": counts["delta1_sum"] / pixels if pixels else 0.0,
                "eval_depth_in": eval_depth_in,
            }
        )
    return rows


def update_road_counts(accumulator, scope, logits, target, threshold):
    valid = target != 255
    prediction = logits[valid].sigmoid() >= threshold
    ground_truth = target[valid] == 1

    accumulator[scope]["valid_pixels"] += int(valid.sum().item())
    accumulator[scope]["tp"] += int((prediction & ground_truth).sum().item())
    accumulator[scope]["fp"] += int((prediction & ~ground_truth).sum().item())
    accumulator[scope]["fn"] += int((~prediction & ground_truth).sum().item())


def finish_road_metrics(accumulator, dates, threshold):
    rows = []
    for scope in ["all", *sorted(dates)]:
        counts = accumulator[scope]
        tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
        precision, recall, f1 = safe_prf(tp, fp, fn)
        rows.append(
            {
                "scope": scope,
                "valid_pixels": int(counts["valid_pixels"]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold": threshold,
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def tensor_to_image(image_tensor):
    array = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).clip(0, 255).astype(np.uint8))


def add_title(image, title):
    font = load_font(20)
    header_height = font.size + 12
    panel = Image.new("RGB", (image.width, image.height + header_height), (20, 20, 20))
    panel.paste(image, (0, header_height))
    ImageDraw.Draw(panel).text((8, 5), title, fill=(255, 255, 255), font=font)
    return panel


def colorize_depth(depth, valid, args, sparse):
    clipped = np.clip(depth, args.min_depth, args.max_depth)
    normalized = ((clipped - args.min_depth) / (args.max_depth - args.min_depth) * 255).astype(np.uint8)
    colored = cv2.cvtColor(
        cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    if sparse:
        colored[~valid] = 0
    return Image.fromarray(colored)


def road_overlay(image, mask, valid, color):
    base = np.asarray(image).copy()
    overlay = base.copy()
    overlay[mask & valid] = np.asarray(color, dtype=np.uint8)
    blended = (0.55 * base + 0.45 * overlay).astype(np.uint8)
    blended[~valid] = (0.35 * base[~valid]).astype(np.uint8)
    return Image.fromarray(blended)


def build_task_panels(
    image_tensor,
    gt_boxes,
    pred_boxes,
    gt_depth,
    pred_depth,
    valid_depth,
    gt_road,
    pred_road_logits,
    args,
):
    """Build separate GT/Pred panels shared by combined and task-wise output."""
    image = tensor_to_image(image_tensor)
    box_font = load_font(12)
    gt_depth_np = gt_depth.detach().cpu().numpy()
    pred_depth_np = pred_depth.detach().cpu().numpy()
    valid_depth_np = valid_depth.detach().cpu().numpy().astype(bool)
    gt_road_np = gt_road.detach().cpu().numpy()
    road_valid = gt_road_np != 255

    return {
        ("detection", "gt"): draw_boxes(
            image, gt_boxes, f"GT DET | {len(gt_boxes)} boxes", box_font
        ),
        ("detection", "pred"): draw_boxes(
            image, pred_boxes, f"PRED DET | {len(pred_boxes)} boxes", box_font
        ),
        ("depth", "gt"): add_title(
            colorize_depth(gt_depth_np, valid_depth_np, args, sparse=True),
            f"GT DEPTH | valid={valid_depth_np.sum()}",
        ),
        ("depth", "pred"): add_title(
            colorize_depth(pred_depth_np, np.ones_like(valid_depth_np), args, sparse=False),
            f"PRED DEPTH | {args.min_depth:g}-{args.max_depth:g} m",
        ),
        ("road", "gt"): add_title(
            road_overlay(image, gt_road_np == 1, road_valid, color=(0, 255, 80)),
            "GT ROAD",
        ),
        ("road", "pred"): add_title(
            road_overlay(
                image,
                pred_road_logits.detach().cpu().sigmoid().numpy() >= args.road_threshold,
                road_valid,
                color=(255, 80, 255),
            ),
            f"PRED ROAD | threshold={args.road_threshold:.2f}",
        ),
    }


def save_task_visualizations(panels, output_root, date, name):
    """Save GT and prediction independently under task-specific folders."""
    for (task, kind), panel in panels.items():
        path = output_root / "task_images" / task / kind / date / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(path, quality=92, subsampling=0)


def save_six_panel(
    image_tensor,
    gt_boxes,
    pred_boxes,
    gt_depth,
    pred_depth,
    valid_depth,
    gt_road,
    pred_road_logits,
    output_path,
    args,
):
    task_panels = build_task_panels(
        image_tensor, gt_boxes, pred_boxes, gt_depth, pred_depth, valid_depth,
        gt_road, pred_road_logits, args,
    )
    panels = [
        task_panels[(task, kind)]
        for task in ("detection", "depth", "road")
        for kind in ("gt", "pred")
    ]
    gap = 8
    panel_width, panel_height = panels[0].size
    canvas = Image.new(
        "RGB",
        (panel_width * 2 + gap, panel_height * 3 + gap * 2),
        (255, 255, 255),
    )
    for index, panel in enumerate(panels):
        x = (index % 2) * (panel_width + gap)
        y = (index // 2) * (panel_height + gap)
        canvas.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0)
    return task_panels


def evenly_spaced(entries, count):
    if len(entries) <= count:
        return entries
    indices = [round(index * (len(entries) - 1) / (count - 1)) for index in range(count)]
    return [entries[index] for index in indices]


def make_overview(entries, output_path, args):
    chosen = evenly_spaced(entries, args.sheet_count)
    columns = 3
    thumb_width = args.thumb_width
    thumb_height = round(thumb_width * 1.5)
    caption_height = 28
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + caption_height)),
        (25, 25, 25),
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)
    for index, entry in enumerate(chosen):
        with Image.open(entry["rendered"]) as source:
            image = source.convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + caption_height)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y))
        draw.text((x + 4, y + thumb_height + 4), entry["name"], fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90, subsampling=0)


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_reports(detection, depth, road, performance):
    print("\nInference performance (model forward only)")
    print(f"images              : {performance['images']}")
    print(f"batches             : {performance['batches']}")
    print(f"total_inference_ms  : {performance['total_inference_ms']:.3f}")
    print(f"inference_ms/image  : {performance['inference_ms']:.3f}")
    print(f"FPS                 : {performance['fps']:.3f}")
    print(f"FLOPs/image         : {performance['gflops']:.3f} GFLOPs")

    print("\nDetection metrics (class-aware)")
    print(f"{'scope':<8} {'class':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>8} {'R':>8} {'F1':>8}")
    for row in detection:
        print(
            f"{row['scope']:<8} {row['class']:<12} {row['tp']:>6} {row['fp']:>6} "
            f"{row['fn']:>6} {row['precision']:>8.4f} {row['recall']:>8.4f} {row['f1']:>8.4f}"
        )

    print("\nDepth metrics (valid GT pixels only)")
    print(f"{'scope':<8} {'pixels':>12} {'AbsRel':>10} {'RMSE':>10} {'delta1':>10}")
    for row in depth:
        print(
            f"{row['scope']:<8} {row['valid_pixels']:>12} {row['abs_rel']:>10.4f} "
            f"{row['rmse']:>10.4f} {row['delta1']:>10.4f}"
        )

    print("\nRoad segmentation metrics")
    print(f"{'scope':<8} {'IoU':>10} {'P':>10} {'R':>10} {'F1':>10}")
    for row in road:
        print(
            f"{row['scope']:<8} {row['iou']:>10.4f} {row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} {row['f1']:>10.4f}"
        )


def html_table(title, rows, columns):
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<h2>{title}</h2><table><tr>{header}</tr>{''.join(body)}</table>"


def save_html(output_root, entries, detection, depth, road, performance, args):
    tables = [
        html_table(
            "Inference performance",
            [performance],
            [("images", "Images"), ("batches", "Batches"),
             ("total_inference_ms", "Total inference (ms)"),
             ("inference_ms", "Inference (ms/image)"),
             ("fps", "FPS"), ("gflops", "GFLOPs/image")],
        ),
        html_table(
            "Detection",
            detection,
            [("scope", "Scope"), ("class", "Class"), ("tp", "TP"), ("fp", "FP"),
             ("fn", "FN"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")],
        ),
        html_table(
            "Depth",
            depth,
            [("scope", "Scope"), ("valid_pixels", "Pixels"), ("abs_rel", "AbsRel"),
             ("rmse", "RMSE"), ("delta1", "delta1")],
        ),
        html_table(
            "Road segmentation",
            road,
            [("scope", "Scope"), ("iou", "IoU"), ("precision", "Precision"),
             ("recall", "Recall"), ("f1", "F1")],
        ),
    ]

    sections = []
    for date in sorted({entry["date"] for entry in entries}):
        cards = []
        for entry in entries:
            if entry["date"] != date or not entry["rendered"]:
                continue
            relative = Path(entry["rendered"]).relative_to(output_root).as_posix()
            cards.append(
                f'<a class="card" href="{html.escape(relative)}">'
                f'<img loading="lazy" src="{html.escape(relative)}">'
                f'<span>{html.escape(entry["name"])}</span></a>'
            )
        sections.append(f'<h2>{date} test</h2><div class="grid">{"".join(cards)}</div>')

    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>YOLO 3-task test</title><style>
body{{background:#171717;color:#eee;font-family:sans-serif;margin:20px}}
table{{border-collapse:collapse;margin:16px 0}}th,td{{border:1px solid #666;padding:6px 10px;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:12px}}
.card{{background:#292929;color:#eee;text-decoration:none;padding:6px}}.card img{{width:100%;display:block}}
.card span{{display:block;padding:5px}}code{{color:#ddd}}
</style></head><body><h1>Detection + Depth + Road Test</h1>
<p>checkpoint: <code>{html.escape(str(Path(args.checkpoint).resolve()))}</code></p>
{"".join(tables)}{"".join(sections)}</body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------


def move_batch_to_device(batch, device):
    for key in (
        "img", "nir", "depth", "valid_depth", "road", "batch_idx", "cls", "bboxes"
    ):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def evaluate(model, loader, device, output_root, args):
    detection_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    depth_counts = defaultdict(lambda: defaultdict(float))
    road_counts = defaultdict(lambda: defaultdict(float))
    entries = []
    processed = 0
    measured_batches = 0
    inference_seconds = 0.0

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            # One shared multi-task forward for all three outputs.
            synchronize(device)
            forward_start = time.perf_counter()
            raw_detection, predicted_depth, road_logits = model(batch["img"])
            synchronize(device)
            inference_seconds += time.perf_counter() - forward_start
            measured_batches += 1
            predicted_boxes = decode_detections(raw_detection, args)

            for sample_index, prediction in enumerate(predicted_boxes):
                name = batch["names"][sample_index]
                date = batch["dates"][sample_index]
                gt_boxes = get_gt_boxes(batch, sample_index, resolve_image_size(args))

                sample_depth = batch["depth"][sample_index]
                sample_valid_depth = batch["valid_depth"][sample_index]
                gt_boxes = filter_boxes_by_gt_depth(
                    gt_boxes, sample_depth, sample_valid_depth, args.eval_depth_in
                )
                prediction = filter_boxes_by_gt_depth(
                    prediction, sample_depth, sample_valid_depth, args.eval_depth_in
                )
                metric_depth_valid = depth_evaluation_mask(
                    sample_depth, sample_valid_depth, args.eval_depth_in
                )

                image_detection_counts = match_detections(gt_boxes, prediction, args.eval_iou)
                for scope in ("all", date):
                    update_detection_counts(detection_counts, scope, image_detection_counts)
                    update_depth_counts(
                        depth_counts,
                        scope,
                        predicted_depth[sample_index],
                        sample_depth,
                        metric_depth_valid,
                    )
                    update_road_counts(
                        road_counts,
                        scope,
                        road_logits[sample_index],
                        batch["road"][sample_index],
                        args.road_threshold,
                    )

                rendered = ""
                if not args.no_images:
                    output_path = output_root / "images" / date / f"{name}.jpg"
                    task_panels = save_six_panel(
                        batch["img"][sample_index],
                        gt_boxes,
                        prediction,
                        batch["depth"][sample_index],
                        predicted_depth[sample_index],
                        batch["valid_depth"][sample_index],
                        batch["road"][sample_index],
                        road_logits[sample_index],
                        output_path,
                        args,
                    )
                    save_task_visualizations(task_panels, output_root, date, name)
                    rendered = str(output_path)

                entries.append({"name": name, "date": date, "rendered": rendered})
                processed += 1
            print(f"processed {processed}/{len(loader.dataset)}")

    dates = {entry["date"] for entry in entries}
    performance = {
        "images": processed,
        "batches": measured_batches,
        "total_inference_ms": inference_seconds * 1000.0,
        "inference_ms": inference_seconds * 1000.0 / max(processed, 1),
        "fps": processed / max(inference_seconds, 1e-12),
    }
    return (
        entries,
        finish_detection_metrics(detection_counts, dates, args),
        finish_depth_metrics(depth_counts, dates, args.eval_depth_in),
        finish_road_metrics(road_counts, dates, args.road_threshold),
        performance,
    )


def main():
    args = parse_args()
    if args.eval_depth_in is not None and args.eval_depth_in <= args.min_depth:
        raise ValueError(
            f"--eval-depth-in must be greater than --min-depth ({args.min_depth} m)"
        )
    print(
        "evaluation range: "
        + (f"GT depth <= {args.eval_depth_in:g} m" if args.eval_depth_in is not None else "full")
    )
    image_size = resolve_image_size(args)
    os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")
    device = select_device(str(args.device))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(args, device)
    dataset = ThreeTaskDataset(
        args.dataset_root,
        split="test",
        size=image_size,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=False,
    )
    if args.limit > 0:
        dataset.rows = dataset.rows[: args.limit]
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(
        f"epoch={checkpoint.get('epoch')} test={len(dataset)} "
        f"input={image_size[0]}x{image_size[1]} device={device}"
    )
    gflops = float(get_flops(model, imgsz=list(image_size)))
    warm_up_model(model, device, image_size, args.warmup_runs)
    entries, detection, depth, road, performance = evaluate(
        model, loader, device, output_root, args
    )
    performance.update(
        {
            "gflops": gflops,
            "image_height": image_size[0],
            "image_width": image_size[1],
            "batch_size": args.batch,
            "device": str(device),
            "warmup_runs": args.warmup_runs,
        }
    )

    save_csv(output_root / "detection_metrics.csv", detection)
    save_csv(output_root / "depth_metrics.csv", depth)
    save_csv(output_root / "road_metrics.csv", road)
    save_csv(output_root / "inference_metrics.csv", [performance])
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "settings": vars(args),
        "detection": detection,
        "depth": depth,
        "road": road,
        "performance": performance,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_images:
        for date in sorted({entry["date"] for entry in entries}):
            make_overview(
                [entry for entry in entries if entry["date"] == date],
                output_root / f"overview_{date}_test_3task.jpg",
                args,
            )
    save_html(output_root, entries, detection, depth, road, performance, args)
    print_reports(detection, depth, road, performance)
    print(f"\noutput: {output_root}")
    print(f"index: {output_root / 'index.html'}")
    print(f"summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
