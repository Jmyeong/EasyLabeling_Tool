#!/usr/bin/env python3
"""Generate DY 4-class detection boxes directly from raw instance masks.

Raw RGB, depth, semantic, and instance files remain in place. Only compact
YOLO labels, bbox JSON, a manifest, and image symlinks are created under
``<raw-root>/yolo_4class_from_instseg``.
"""

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = ["Golfcart", "Person", "Tree", "Undef_obj"]
RGB_TO_SOURCE_CLASS = {
    (96, 96, 192): 2,
    (192, 96, 96): 2,
    (160, 0, 128): 3,
    (128, 0, 160): 3,
    (0, 96, 144): 4,
    (144, 96, 0): 4,
    (172, 233, 115): 7,
    (249, 25, 35): 7,
}
SOURCE_TO_YOLO = {3: 0, 2: 1, 4: 2, 7: 3}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create four-class YOLO boxes from raw DY instance masks"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <raw-root>/yolo_4class_from_instseg",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help=(
            "Optional JSON: {date: {train: [sequence...], test: [sequence...]}}. "
            "Default uses every sequence except the last for train."
        ),
    )
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def find_image(directory, stem):
    for suffix in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def index_pngs(directory):
    indexed = {}
    if directory.is_dir():
        for path in directory.rglob("*.png"):
            # Ignore Windows alternate-stream metadata extracted as normal files.
            if "Zone.Identifier" not in path.name:
                indexed.setdefault(path.stem, path)
    return indexed


def has_raw_case(case_dir):
    return (
        (case_dir / "images").is_dir()
        and (case_dir / "depth_gt/depth_npy").is_dir()
        and (case_dir / "segmentation/SegmentationClass").is_dir()
        and (case_dir / "segmentation/SegmentationObject").is_dir()
    )


def discover_cases(raw_root):
    datasets = {}
    for date_dir in sorted(raw_root.iterdir()):
        if not date_dir.is_dir() or not re.fullmatch(r"\d{6}", date_dir.name):
            continue
        cases = sorted(
            (path for path in date_dir.iterdir() if path.is_dir() and has_raw_case(path)),
            key=lambda path: natural_key(path.name),
        )
        if cases:
            datasets[date_dir.name] = cases
    if not datasets:
        raise RuntimeError(f"No raw DY cases found under {raw_root}")
    return datasets


def resolve_splits(datasets, config_path):
    configured = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    result = {}
    for date, case_paths in datasets.items():
        names = [path.name for path in case_paths]
        if date in configured:
            train = list(configured[date].get("train", []))
            test = list(configured[date].get("test", []))
        else:
            if len(names) < 2:
                raise ValueError(f"{date} has fewer than two cases; provide --split-config")
            train, test = names[:-1], [names[-1]]
        unknown = sorted(set(train + test) - set(names))
        overlap = sorted(set(train) & set(test))
        unassigned = sorted(set(names) - set(train) - set(test))
        if unknown or overlap or unassigned or not train or not test:
            raise ValueError(
                f"Invalid split for {date}: unknown={unknown}, overlap={overlap}, "
                f"unassigned={unassigned}"
            )
        result[date] = {**{name: "train" for name in train}, **{name: "test" for name in test}}
    return result


def align_label(path, image_size):
    label = Image.open(path).convert("RGB")
    width, height = image_size
    if label.size == (width, height * 2):
        # Some exported labels contain an identifier view above the actual mask.
        label = label.crop((0, height, width, height * 2))
    if label.size != image_size:
        label = label.resize(image_size, Image.Resampling.NEAREST)
    return np.asarray(label)


def semantic_class_mask(semantic_rgb):
    result = np.full(semantic_rgb.shape[:2], -1, dtype=np.int16)
    for color, source_class in RGB_TO_SOURCE_CLASS.items():
        result[np.all(semantic_rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = source_class
    return result


def majority_source_class(instance_mask, class_mask):
    values = class_mask[instance_mask]
    values = values[values >= 0]
    if values.size == 0:
        return None
    classes, counts = np.unique(values, return_counts=True)
    return int(classes[counts.argmax()])


def boxes_from_instance(instance_rgb, semantic_rgb, min_area):
    class_mask = semantic_class_mask(semantic_rgb)
    object_region = class_mask >= 0
    boxes = []
    encoded = (
        instance_rgb[:, :, 0].astype(np.uint32) << 16
        | instance_rgb[:, :, 1].astype(np.uint32) << 8
        | instance_rgb[:, :, 2].astype(np.uint32)
    )
    for color_value in np.unique(encoded[object_region]):
        color_tuple = (
            int((color_value >> 16) & 255),
            int((color_value >> 8) & 255),
            int(color_value & 255),
        )
        if color_tuple == (0, 0, 0):
            continue
        instance_mask = (encoded == color_value) & object_region
        area = int(instance_mask.sum())
        if area < min_area:
            continue
        source_class = majority_source_class(instance_mask, class_mask)
        if source_class not in SOURCE_TO_YOLO:
            continue
        ys, xs = np.where(instance_mask)
        boxes.append(
            {
                "class_id": SOURCE_TO_YOLO[source_class],
                "class_name": CLASS_NAMES[SOURCE_TO_YOLO[source_class]],
                "source_class_id": source_class,
                "instance_color": list(color_tuple),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ],
                "area": area,
            }
        )
    return sorted(boxes, key=lambda box: (box["class_id"], box["bbox_xyxy"]))


def yolo_line(box, width, height):
    x1, y1, x2, y2 = box["bbox_xyxy"]
    center_x = (x1 + x2) / (2.0 * width)
    center_y = (y1 + y2) / (2.0 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return (
        f"{box['class_id']} {center_x:.8f} {center_y:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )


def safe_symlink(source, destination):
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Generated image target already exists: {destination}")
    os.symlink(source.resolve(), destination)


def write_data_yaml(path, output_root, train_list, val_list):
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    path.write_text(
        f"path: {output_root}\ntrain: {train_list}\nval: {val_list}\nnames:\n{names}\n",
        encoding="utf-8",
    )


def clean_generated(output_root):
    for name in ("images", "labels", "bboxes"):
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
    for pattern in ("*.txt", "*.yaml", "manifest.csv", "summary.json"):
        for path in output_root.glob(pattern):
            path.unlink()


def main():
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else raw_root / "yolo_4class_from_instseg"
    )
    if args.min_area < 1:
        raise ValueError("--min-area must be at least 1")
    if args.clean and output_root.exists():
        clean_generated(output_root)

    datasets = discover_cases(raw_root)
    splits = resolve_splits(datasets, args.split_config)
    for split in ("train", "test"):
        for name in ("images", "labels", "bboxes"):
            (output_root / name / split).mkdir(parents=True, exist_ok=True)

    rows = []
    image_lists = defaultdict(list)
    sample_counts = Counter()
    box_counts = Counter()
    empty_counts = Counter()
    missing_counts = Counter()

    for date, case_paths in datasets.items():
        for case_dir in case_paths:
            split = splits[date][case_dir.name]
            semantic_index = index_pngs(case_dir / "segmentation/SegmentationClass")
            instance_index = index_pngs(case_dir / "segmentation/SegmentationObject")
            image_paths = sorted(
                (
                    path
                    for path in (case_dir / "images").iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda path: natural_key(path.stem),
            )
            for image_path in image_paths:
                stem = image_path.stem
                depth_path = case_dir / "depth_gt/depth_npy" / f"{stem}.npy"
                semantic_path = semantic_index.get(stem)
                instance_path = instance_index.get(stem)
                missing = [
                    name
                    for name, path in (
                        ("depth", depth_path if depth_path.is_file() else None),
                        ("semantic", semantic_path),
                        ("instance", instance_path),
                    )
                    if path is None
                ]
                if missing:
                    for name in missing:
                        missing_counts[(date, case_dir.name, name)] += 1
                    continue

                with Image.open(image_path) as image:
                    image_size = image.size
                semantic_rgb = align_label(semantic_path, image_size)
                instance_rgb = align_label(instance_path, image_size)
                boxes = boxes_from_instance(instance_rgb, semantic_rgb, args.min_area)

                name = f"golf_cart_{date}_{case_dir.name}_{stem}"
                yolo_image = output_root / "images" / split / f"{name}{image_path.suffix.lower()}"
                label_path = output_root / "labels" / split / f"{name}.txt"
                bbox_path = output_root / "bboxes" / split / f"{name}.json"
                safe_symlink(image_path, yolo_image)
                width, height = image_size
                label_path.write_text(
                    "\n".join(yolo_line(box, width, height) for box in boxes)
                    + ("\n" if boxes else ""),
                    encoding="utf-8",
                )
                bbox_path.write_text(
                    json.dumps(
                        {
                            "name": name,
                            "image_size": [width, height],
                            "semantic": str(semantic_path.resolve()),
                            "instance": str(instance_path.resolve()),
                            "bboxes": boxes,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                image_lists[split].append(str(yolo_image.absolute()))
                if split == "test":
                    image_lists[f"test_{date}"].append(str(yolo_image.absolute()))
                sample_counts[(split, date)] += 1
                if not boxes:
                    empty_counts[(split, date)] += 1
                for box in boxes:
                    box_counts[(split, date, box["class_name"])] += 1
                rows.append(
                    {
                        "name": name,
                        "date": date,
                        "split": split,
                        "case": case_dir.name,
                        "frame": stem,
                        "image": str(image_path.resolve()),
                        "source_image": str(image_path.resolve()),
                        "depth_npy": str(depth_path.resolve()),
                        "semseg": str(semantic_path.resolve()),
                        "instseg": str(instance_path.resolve()),
                        "label": str(label_path.resolve()),
                        "bbox_json": str(bbox_path.resolve()),
                        "yolo_image": str(yolo_image.absolute()),
                        "boxes": len(boxes),
                    }
                )

    if not rows:
        raise RuntimeError("No complete raw samples were found")
    for key, paths in image_lists.items():
        (output_root / f"{key}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    write_data_yaml(output_root / "data.yaml", output_root, "train.txt", "test.txt")
    for date in sorted(datasets):
        write_data_yaml(
            output_root / f"data_{date}.yaml",
            output_root,
            "train.txt",
            f"test_{date}.txt",
        )
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "classes": CLASS_NAMES,
        "splits": {
            date: {
                split: sorted(case for case, value in mapping.items() if value == split)
                for split in ("train", "test")
            }
            for date, mapping in splits.items()
        },
        "samples": {"/".join(key): value for key, value in sorted(sample_counts.items())},
        "empty_samples": {"/".join(key): value for key, value in sorted(empty_counts.items())},
        "boxes": {"/".join(key): value for key, value in sorted(box_counts.items())},
        "missing": {"/".join(key): value for key, value in sorted(missing_counts.items())},
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
