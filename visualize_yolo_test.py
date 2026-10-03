#!/usr/bin/env python3
"""Visualize YOLO test predictions beside ground truth boxes."""

import argparse
import csv
import html
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


CLASS_NAMES = ["Golfcart", "Person", "Tree", "Undef_obj"]
COLORS = [(255, 80, 50), (30, 220, 255), (60, 220, 80), (220, 70, 255)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="./runs/"
        "yolo11s_DY260618_260724_4class_bboxfixed/weights/best.pt",
    )
    parser.add_argument(
        "--dataset-root",
        default="./datasets/DY_260618_260724_4class",
    )
    parser.add_argument(
        "--output-root",
        default="./test_visualization_gt_pred",
    )
    parser.add_argument("--device", default="4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold")
    parser.add_argument(
        "--eval-iou",
        type=float,
        default=0.50,
        help="Class-aware TP matching IoU threshold",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sheet-count", type=int, default=48)
    parser.add_argument("--thumb-width", type=int, default=480)
    return parser.parse_args()


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_gt(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, bw, bh = map(float, line.split())
        boxes.append(
            (
                int(class_id),
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
                None,
            )
        )
    return boxes


def read_predictions(result):
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    return [
        (int(class_id), *coords, float(confidence))
        for coords, class_id, confidence in zip(xyxy, classes, confidences)
    ]


def box_iou(box_a, box_b):
    x1 = max(box_a[1], box_b[1])
    y1 = max(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])
    y2 = min(box_a[4], box_b[4])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[3] - box_a[1]) * max(0.0, box_a[4] - box_a[2])
    area_b = max(0.0, box_b[3] - box_b[1]) * max(0.0, box_b[4] - box_b[2])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(gt, pred, iou_threshold):
    """Greedily match confidence-sorted predictions to same-class GT once."""
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(len(CLASS_NAMES))}
    for class_id in range(len(CLASS_NAMES)):
        class_gt = [box for box in gt if box[0] == class_id]
        class_pred = sorted(
            (box for box in pred if box[0] == class_id),
            key=lambda box: box[5],
            reverse=True,
        )
        unmatched = set(range(len(class_gt)))
        for prediction in class_pred:
            candidates = [(box_iou(prediction, class_gt[index]), index) for index in unmatched]
            best_iou, best_index = max(candidates, default=(0.0, None))
            if best_index is not None and best_iou >= iou_threshold:
                counts[class_id]["tp"] += 1
                unmatched.remove(best_index)
            else:
                counts[class_id]["fp"] += 1
        counts[class_id]["fn"] += len(unmatched)
    return counts


def metric_values(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def build_metrics(accumulator, dates, conf, eval_iou):
    rows = []
    for scope in ["all", *sorted(dates)]:
        class_counts = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            counts = accumulator[(scope, class_id)]
            precision, recall, f1 = metric_values(counts["tp"], counts["fp"], counts["fn"])
            class_counts.append(counts)
            rows.append(
                {
                    "scope": scope,
                    "class": class_name,
                    **counts,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confidence": conf,
                    "eval_iou": eval_iou,
                }
            )
        total = {
            key: sum(counts[key] for counts in class_counts) for key in ("tp", "fp", "fn")
        }
        precision, recall, f1 = metric_values(total["tp"], total["fp"], total["fn"])
        rows.insert(
            len(rows) - len(CLASS_NAMES),
            {
                "scope": scope,
                "class": "All(micro)",
                **total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confidence": conf,
                "eval_iou": eval_iou,
            },
        )
    return rows


def save_metrics(metrics, output_root, model_path, conf, nms_iou, eval_iou):
    fieldnames = [
        "scope", "class", "tp", "fp", "fn", "precision", "recall", "f1",
        "confidence", "eval_iou",
    ]
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    payload = {
        "model": str(model_path),
        "matching": "class-aware, confidence-descending greedy one-to-one",
        "confidence_threshold": conf,
        "nms_iou": nms_iou,
        "eval_iou": eval_iou,
        "metrics": metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_metrics(metrics):
    print("\nDetection metrics (class-aware)")
    print(f"{'scope':<8} {'class':<12} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for row in metrics:
        print(
            f"{row['scope']:<8} {row['class']:<12} {row['tp']:>5} {row['fp']:>5} "
            f"{row['fn']:>5} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}"
        )


def draw_label(draw, x, y, text, color, font):
    x, y = max(1, x), max(1, y)
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=color)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def draw_boxes(image, boxes, heading, font):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 180))
    for class_id, x1, y1, x2, y2, confidence in sorted(
        boxes, key=lambda box: (box[3] - box[1]) * (box[4] - box[2]), reverse=True
    ):
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue
        color = COLORS[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = CLASS_NAMES[class_id]
        if confidence is not None:
            text += f" {confidence:.2f}"
        draw_label(draw, x1 + 2, max(1, y1 - font.size - 3), text, color, font)

    header_font = load_font(max(18, round(image.height / 25)))
    header_height = header_font.size + 12
    panel = Image.new("RGB", (image.width, image.height + header_height), (20, 20, 20))
    panel.paste(image, (0, header_height))
    ImageDraw.Draw(panel).text((8, 5), heading, fill=(255, 255, 255), font=header_font)
    return panel


def save_comparison(row, gt, pred, output_path, font):
    with Image.open(row["image"]) as source:
        image = source.convert("RGB")

    left = draw_boxes(image, gt, f"GT | boxes={len(gt)}", font)
    right = draw_boxes(image, pred, f"PRED | boxes={len(pred)}", font)
    gap = 8
    comparison = Image.new("RGB", (left.width + right.width + gap, left.height), (255, 255, 255))
    comparison.paste(left, (0, 0))
    comparison.paste(right, (left.width + gap, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path, quality=92, subsampling=0)
    return len(gt), len(pred)


def evenly_spaced(rows, count):
    if len(rows) <= count:
        return rows
    indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[i] for i in indices]


def make_sheet(entries, path, thumb_width, count, font):
    chosen = evenly_spaced(entries, count)
    columns = 3
    thumb_height = round(thumb_width * 0.31)
    caption_height = 30
    row_count = (len(chosen) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumb_width, row_count * (thumb_height + caption_height)), (25, 25, 25)
    )
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(chosen):
        with Image.open(entry["rendered"]) as source:
            image = source.convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = index % columns * thumb_width
        y = index // columns * (thumb_height + caption_height)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y))
        draw.text(
            (x + 4, y + thumb_height + 5),
            f"{entry['name']}  GT={entry['gt']} PRED={entry['pred']}",
            fill=(240, 240, 240),
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90, subsampling=0)


def make_html(entries, metrics, output_root, model_path, conf, eval_iou):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["date"]].append(entry)
    sections = []
    for date, group in sorted(groups.items()):
        cards = []
        for entry in group:
            relative = Path(entry["rendered"]).relative_to(output_root).as_posix()
            cards.append(
                f'<a class="card" href="{html.escape(relative)}">'
                f'<img loading="lazy" src="{html.escape(relative)}">'
                f'<span>{html.escape(entry["name"])} | GT={entry["gt"]} '
                f'PRED={entry["pred"]}</span></a>'
            )
        sections.append(
            f'<h2>{date} test ({len(group)} images)</h2><div class="grid">{"".join(cards)}</div>'
        )
    legend = " ".join(
        f'<span style="color:rgb{COLORS[i]}">{name}</span>' for i, name in enumerate(CLASS_NAMES)
    )
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['scope'])}</td><td>{html.escape(row['class'])}</td>"
        f"<td>{row['tp']}</td><td>{row['fp']}</td><td>{row['fn']}</td>"
        f"<td>{row['precision']:.4f}</td><td>{row['recall']:.4f}</td>"
        f"<td>{row['f1']:.4f}</td></tr>"
        for row in metrics
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>YOLO test GT vs prediction</title><style>
body {{ background:#171717;color:#eee;font-family:sans-serif;margin:20px }}
.grid {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:12px }}
.card {{ background:#292929;color:#eee;text-decoration:none;padding:6px }}
.card img {{ width:100%;display:block }} .card span {{ display:block;padding:6px }}
.legend span {{ font-weight:bold;margin-right:18px }} code {{ color:#ddd }}
table {{ border-collapse:collapse;margin:18px 0 }} th,td {{ border:1px solid #666;padding:6px 10px;text-align:right }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left }}
</style></head><body><h1>YOLO Test: GT (left) vs PRED (right)</h1>
<p>model: <code>{html.escape(str(model_path))}</code><br>confidence threshold: {conf};
class-aware evaluation IoU: {eval_iou}</p>
<div class="legend">{legend}</div>
<table><thead><tr><th>Scope</th><th>Class</th><th>TP</th><th>FP</th><th>FN</th>
<th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{metric_rows}</tbody></table>
{''.join(sections)}</body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")

    with (dataset_root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    image_paths = [row["image"] for row in rows]
    model = YOLO(str(model_path))
    results = model.predict(
        source=image_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        stream=False,
        verbose=False,
        save=False,
    )
    if len(results) != len(rows):
        raise RuntimeError(f"prediction count mismatch: {len(results)} != {len(rows)}")

    box_font = load_font(12)
    sheet_font = load_font(13)
    entries = []
    accumulator = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for index, (row, result) in enumerate(zip(rows, results), 1):
        height, width = result.orig_shape
        gt = read_gt(Path(row["label"]), width, height)
        pred = read_predictions(result)
        image_counts = match_detections(gt, pred, args.eval_iou)
        for class_id, counts in image_counts.items():
            for scope in ("all", row["date"]):
                for key in ("tp", "fp", "fn"):
                    accumulator[(scope, class_id)][key] += counts[key]
        output_path = output_root / "images" / row["date"] / f"{row['name']}.jpg"
        gt_count, pred_count = save_comparison(row, gt, pred, output_path, box_font)
        entries.append(
            {**row, "rendered": str(output_path), "gt": gt_count, "pred": pred_count}
        )
        if index % 50 == 0 or index == len(rows):
            print(f"rendered {index}/{len(rows)}")

    for date in sorted({entry["date"] for entry in entries}):
        group = [entry for entry in entries if entry["date"] == date]
        make_sheet(
            group,
            output_root / f"overview_{date}_test_gt_pred.jpg",
            args.thumb_width,
            args.sheet_count,
            sheet_font,
        )
    metrics = build_metrics(
        accumulator, {entry["date"] for entry in entries}, args.conf, args.eval_iou
    )
    save_metrics(metrics, output_root, model_path, args.conf, args.iou, args.eval_iou)
    make_html(entries, metrics, output_root, model_path, args.conf, args.eval_iou)
    print_metrics(metrics)
    print(f"output: {output_root}")
    print(f"index: {output_root / 'index.html'}")
    print(f"metrics: {output_root / 'metrics.csv'}")


if __name__ == "__main__":
    main()
