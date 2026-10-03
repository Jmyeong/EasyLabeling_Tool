"""Adapter for DY/datasets manifests and separately stored review annotations."""

import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from prepare_raw_dy_from_instseg import align_label, boxes_from_instance


def add_dataset_arguments(parser):
    parser.add_argument("--pseudo-root", default="", help="Default: <dataset-root>/pseudo_labels")
    parser.add_argument(
        "--pseudo-label-policy", choices=("exclude", "reviewed", "all"), default="reviewed",
        help="all includes pending pseudo labels in TRAIN only; evaluation always requires approval",
    )
    parser.add_argument(
        "--depth-source", choices=("completed", "filtered", "raw"), default="completed",
        help="Completed depth includes estimates in train; evaluation uses only source==1",
    )


def dataset_options(args):
    return {key: getattr(args, key) for key in ("pseudo_root", "pseudo_label_policy", "depth_source")}


def read_manifest(root):
    with (Path(root) / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_file(root, value):
    return str((Path(root) / value.replace("\\", "/")).resolve()) if value else ""


def annotation_paths(row):
    base = Path(row["date"]) / row["split"] / row["sequence"]
    stem = Path(row["image_file"]).stem
    return {
        "detection_file": (base / "labels" / f"{stem}.txt").as_posix(),
        "road_file": (base / "road" / f"{stem}.png").as_posix(),
        "road_probability_file": (base / "road_probability" / f"{stem}.png").as_posix(),
        "annotation_file": (base / "annotations" / f"{stem}.json").as_posix(),
        "preview_file": (base / "previews" / f"{stem}.jpg").as_posix(),
    }


def adapt_rows(root, rows, split, pseudo_root, policy, depth_source):
    """Never interpret an absent label as a negative detection example."""
    pseudo_root = Path(pseudo_root).expanduser().resolve() if pseudo_root else root / "pseudo_labels"
    output, skipped, sources = [], Counter(), Counter()
    depth_key = {"completed": "completed_depth_file", "filtered": "filtered_depth_file", "raw": "depth_file"}[depth_source]
    for source in rows:
        row = dict(source)
        row.update(
            name=f"{source['date']}_{source['split']}_{source['sequence']}_{Path(source['image_file']).stem}",
            image=resolve_file(root, source["image_file"]),
            nir=resolve_file(root, source["nir_file"]),
            depth_npy=resolve_file(root, source[depth_key]),
            semseg=resolve_file(root, source["semseg_file"]),
            instseg=resolve_file(root, source["instseg_file"]),
            label="", road_mask="", label_source="ground_truth",
        )
        if not (row["semseg"] and row["instseg"]):
            if policy == "exclude":
                skipped["pseudo_excluded"] += 1
                continue
            paths = annotation_paths(source)
            annotation = pseudo_root / paths["annotation_file"]
            if not annotation.is_file():
                skipped["missing_annotation"] += 1
                continue
            record = json.loads(annotation.read_text(encoding="utf-8"))
            if record["image_file"] != source["image_file"]:
                raise ValueError(f"Annotation/image mismatch: {annotation}")
            if record.get("image_sha256") != source.get("image_sha256"):
                raise ValueError(f"Annotation/image hash mismatch: {annotation}")
            status = record.get("review_status", "pending")
            if status != "approved" and not (split == "train" and policy == "all" and status == "pending"):
                skipped[f"review_{status}"] += 1
                continue
            row.update(
                label=resolve_file(pseudo_root, paths["detection_file"]),
                road_mask=resolve_file(pseudo_root, paths["road_file"]),
                label_source="reviewed" if status == "approved" else "pseudo",
            )
        for key in ("image", "nir", "depth_npy", "semseg", "instseg", "label", "road_mask"):
            if row[key] and not Path(row[key]).is_file():
                raise FileNotFoundError(f"{key} missing for {row['name']}: {row[key]}")
        if not all(row[key] for key in ("image", "nir", "depth_npy")):
            raise ValueError(f"Missing RGB/NIR/depth path: {row['name']}")
        output.append(row)
        sources[row["label_source"]] += 1
    print(f"Dated dataset {split}: loaded={len(output)}, labels={dict(sources)}, skipped={dict(skipped)}")
    if not output:
        raise ValueError(f"No labeled {split} samples available; generate/review pseudo labels first")
    return output


@lru_cache(maxsize=2048)
def instance_boxes(semantic_path, instance_path, width, height):
    boxes = boxes_from_instance(
        align_label(instance_path, (width, height)),
        align_label(semantic_path, (width, height)), min_area=1,
    )
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        result.append((box["class_id"], (x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height))
    return result


def read_depth(path, measured_only=False):
    if Path(path).suffix.lower() == ".npz":
        with np.load(path) as data:
            depth = data["depth_m"].astype(np.float32)
            if measured_only and "source" in data:
                depth[data["source"] != 1] = 0
    else:
        depth = np.load(path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    return depth
