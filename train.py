import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace
from dated_dataset import adapt_rows, instance_boxes, read_depth
from prepare_raw_dy_from_instseg import align_label

ROAD_COLORS = {(32, 224, 224), (83, 68, 55)}
KNOWN_COLORS = {
    (0, 0, 0), (83, 68, 55), (32, 224, 224), (96, 96, 192),
    (192, 96, 96), (160, 0, 128), (128, 0, 160), (0, 96, 144),
    (144, 96, 0), (25, 153, 248), (255, 126, 0), (22, 248, 228),
    (172, 233, 115), (249, 25, 35), (245, 61, 224), (16, 160, 192),
    (228, 225, 147), (151, 77, 62),
}
DETECTION_NAMES = {0: "Golfcart", 1: "Person", 2: "Tree", 3: "Undef_obj"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="checkpoints/yolo11s.pt",
        help="YOLO11 checkpoint used to initialize the four-class MTL detector",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="./runs/yolo11s_mtl",
        help="Directory for checkpoints, arguments, and training history",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--height", type=int, default=352, help="Network input height")
    parser.add_argument("--width", type=int, default=640, help="Network input width")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Backward-compatible alias for --width (height remains --height)",
    )
    parser.add_argument("--device", default="4")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--depth-lr-mult", type=float, default=10.0)
    parser.add_argument("--dense-lr-mult", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth-weight", type=float, default=0.20)
    parser.add_argument("--road-weight", type=float, default=0.50)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default="", help="Joint last.pt checkpoint")
    parser.add_argument("--mtl-init", default="", help="Optional 2-task checkpoint")
    return parser.parse_args()


def normalize_image_size(size):
    """Convert a scalar or two-element size into ``(height, width)``."""
    if isinstance(size, (int, float)):
        return int(size), int(size)
    height, width = size
    return int(height), int(width)


def resolve_image_size(args):
    """Resolve CLI arguments into the network input ``(height, width)``."""
    height = args.height
    width = args.imgsz if args.imgsz is not None else args.width
    height, width = normalize_image_size((height, width))
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {height}x{width}")
    return height, width


def collate(samples):
    batch_indices, classes, boxes = [], [], []
    for index, sample in enumerate(samples):
        for cls, cx, cy, bw, bh in sample["boxes"]:
            batch_indices.append(index); classes.append([cls]); boxes.append([cx, cy, bw, bh])
    return {
        "img": torch.stack([sample["img"] for sample in samples]),
        "nir": torch.stack([sample["nir"] for sample in samples]),
        "depth": torch.stack([sample["depth"] for sample in samples]),
        "valid_depth": torch.stack([sample["valid_depth"] for sample in samples]),
        "road": torch.stack([sample["road"] for sample in samples]),
        "batch_idx": torch.tensor(batch_indices, dtype=torch.long),
        "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "dates": [sample["date"] for sample in samples],
        "names": [sample["name"] for sample in samples],
        "nir_names": [sample["nir_name"] for sample in samples],
    }


class Block(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

class DenseDepthRoadHead(nn.Module):
    def __init__(self, min_depth=1.0):
        super().__init__()
        self.min_depth = float(min_depth)
        self.p2 = Block(128, 48)
        self.p3 = Block(128, 64)
        self.p4 = Block(256, 64)
        self.p5 = Block(512, 64)
        self.context = Block(192, 96)
        self.refine = nn.Sequential(Block(144, 96), Block(96, 64))
        self.depth = nn.Sequential(Block(64, 32), nn.Conv2d(32, 1, 1))
        self.road = nn.Sequential(Block(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, features, output_size):
        p2, p3, p4, p5 = features
        p3_size = p3.shape[-2:]
        context = self.context(torch.cat([
            self.p3(p3),
            F.interpolate(self.p4(p4), p3_size, mode="bilinear", align_corners=False),
            F.interpolate(self.p5(p5), p3_size, mode="bilinear", align_corners=False),
        ], dim=1))
        context = F.interpolate(context, p2.shape[-2:], mode="bilinear", align_corners=False)
        dense = self.refine(torch.cat([self.p2(p2), context], dim=1))
        depth = F.interpolate(self.depth(dense), output_size, mode="bilinear", align_corners=False)[:, 0]
        road = F.interpolate(self.road(dense), output_size, mode="bilinear", align_corners=False)[:, 0]
        return F.softplus(depth) + self.min_depth, road



class YoloDepthDetRoad(nn.Module):
    FEATURE_INDICES = (2, 16, 19, 22)

    def __init__(self, detector, min_depth):
        super().__init__()
        self.detector = detector
        self.dense_head = DenseDepthRoadHead(min_depth)

    def forward(self, image):
        saved, features, x = [], {}, image
        for module in self.detector.model:
            if module.f != -1: # module.f : 현재 layer의 입력 출처
                x = saved[module.f] if isinstance(module.f, int) else [
                    x if index == -1 else saved[index] for index in module.f
                ]
            x = module(x)
            saved.append(x if module.i in self.detector.save else None)
            if module.i in self.FEATURE_INDICES:
                features[module.i] = x # depth, semseg용 feature 보관
        depth, road = self.dense_head(tuple(features[index] for index in self.FEATURE_INDICES), image.shape[-2:])
        return x, depth, road


def rgb_to_road(rgb):
    output = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color in KNOWN_COLORS:
        mask = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        output[mask] = 1 if color in ROAD_COLORS else 0
    return output


def letterbox_all(image, nir, depth, road, boxes, size):
    target_height, target_width = normalize_image_size(size)
    height, width = image.shape[:2]
    scale = min(target_height / height, target_width / width)
    new_width, new_height = round(width * scale), round(height * scale)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    out_image = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    out_nir = np.zeros((target_height, target_width), dtype=np.uint8)
    out_depth = np.zeros((target_height, target_width), dtype=np.float32)
    out_road = np.full((target_height, target_width), 255, dtype=np.uint8)
    region = np.s_[pad_y : pad_y + new_height, pad_x : pad_x + new_width]
    out_image[region] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # The synchronized NIR sensor stores 848x480 frames while RGB is 1280x720.
    # Resize it into the RGB coordinate system before applying the exact same
    # letterbox transform so corresponding content stays pixel aligned.
    nir = cv2.resize(nir, (width, height), interpolation=cv2.INTER_LINEAR)
    out_nir[region] = cv2.resize(nir, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    out_depth[region] = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    out_road[region] = cv2.resize(road, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    transformed = []
    for class_id, cx, cy, bw, bh in boxes:
        transformed.append(
            (
                class_id,
                (cx * width * scale + pad_x) / target_width,
                (cy * height * scale + pad_y) / target_height,
                bw * width * scale / target_width,
                bh * height * scale / target_height,
            )
        )
    return out_image, out_nir, out_depth, out_road, transformed

def resolve_dataset_manifest_root(root):
    """Resolve either a prepared manifest root or the raw DY dataset root."""
    root = Path(root).expanduser().resolve()
    if (root / "manifest.csv").is_file():
        return root
    generated = root / "yolo_4class_from_instseg"
    if (generated / "manifest.csv").is_file():
        return generated
    raise FileNotFoundError(
        f"No manifest found under {root}. Generate raw annotations with "
        "prepare_raw_dy_from_instseg.py first."
    )


class ThreeTaskDataset(Dataset):
    def __init__(self, root, split, size, min_depth, max_depth, augment,
                 pseudo_root="", pseudo_label_policy="reviewed", depth_source="completed"):
        self.raw_root = Path(root).expanduser().resolve()
        self.root = resolve_dataset_manifest_root(self.raw_root)
        with (self.root / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        self.size = normalize_image_size(size)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.augment = bool(augment)
        # Evaluation can set a non-zero cyclic offset to pair each RGB sample
        # with a genuinely different NIR frame for causal ablation.
        self.nir_index_offset = 0
        self.nir_pair_indices = None
        self.measured_depth_only = split != "train"
        self.dated_format = bool(self.rows and "image_file" in self.rows[0])
        if self.dated_format:
            self.rows = adapt_rows(self.root, self.rows, split, pseudo_root,
                                   pseudo_label_policy, depth_source)
            return

        needs_cc_lookup = any(
            not row.get("depth_npy") or not row.get("semseg") for row in self.rows
        )
        source_rows = {}
        if needs_cc_lookup:
            for date in sorted({row["date"] for row in self.rows}):
                manifest = Path(
                    f"datasets/{date}_cc_format/manifest.csv"
                )
                with manifest.open(newline="", encoding="utf-8") as handle:
                    for source in csv.DictReader(handle):
                        source_rows[str(Path(source["image"]).resolve())] = source
        for row in self.rows:
            if not row.get("depth_npy") or not row.get("semseg"):
                source = source_rows.get(str(Path(row["source_image"]).resolve()))
                if source is None:
                    raise KeyError(f"CC source row not found for {row['source_image']}")
                row["depth_npy"] = source["depth_npy"]
                row["semseg"] = source["semseg"]
            for key in ("depth_npy", "semseg"):
                if not row[key] or not Path(row[key]).exists():
                    raise FileNotFoundError(f"{key} missing for {row['name']}: {row[key]}")
            image_path = Path(row["image"])
            try:
                image_dir_index = image_path.parts.index("images")
            except ValueError as error:
                raise ValueError(f"Cannot derive NIR path from RGB path: {image_path}") from error
            nir_parts = list(image_path.parts)
            nir_parts[image_dir_index] = "nir"
            row["nir"] = str(Path(*nir_parts))
            if not Path(row["nir"]).is_file():
                raise FileNotFoundError(f"nir missing for {row['name']}: {row['nir']}")

    def __len__(self):
        return len(self.rows)

    def set_grouped_nir_shuffle(self):
        """Pair NIR within the same date and sequence, never across domains."""
        groups = {}
        for index, row in enumerate(self.rows):
            # Frame names end in an integer. Removing it preserves the capture
            # sequence while preventing cross-date/exposure-domain pairing.
            sequence = re.sub(r"_\d+$", "", row["name"])
            groups.setdefault((row["date"], sequence), []).append(index)

        mapping = list(range(len(self.rows)))
        shuffled = 0
        for indices in groups.values():
            if len(indices) < 2:
                continue
            offset = max(1, len(indices) // 2)
            for position, index in enumerate(indices):
                mapping[index] = indices[(position + offset) % len(indices)]
                shuffled += 1
        if shuffled < 2:
            raise ValueError(
                "Grouped NIR shuffle requires at least two samples from one sequence"
            )
        self.nir_pair_indices = mapping
        return shuffled, len(groups)

    @staticmethod
    def read_boxes(path):
        boxes = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls, cx, cy, bw, bh = map(float, line.split())
                boxes.append((int(cls), cx, cy, bw, bh))
        return boxes

    def __getitem__(self, index):
        row = self.rows[index]
        nir_index = (
            self.nir_pair_indices[index]
            if self.nir_pair_indices is not None
            else (index + self.nir_index_offset) % len(self.rows)
        )
        nir_row = self.rows[nir_index]
        image = np.asarray(Image.open(row["image"]).convert("RGB"))
        nir = np.asarray(Image.open(nir_row["nir"]).convert("L"))
        depth = read_depth(row["depth_npy"], measured_only=self.measured_depth_only)
        height, width = image.shape[:2]
        if depth.shape != (height, width):
            raise ValueError(f"Depth/RGB shape mismatch: {row['name']}")
        if row.get("road_mask"):
            road = np.asarray(Image.open(row["road_mask"]))
            if road.shape != (height, width) or not np.isin(road, [0, 1, 255]).all():
                raise ValueError(f"Expected full-resolution 0/1/255 road mask: {row['road_mask']}")
        else:
            road = rgb_to_road(align_label(row["semseg"], (width, height)))
        boxes = (self.read_boxes(row["label"]) if row["label"] else
                 instance_boxes(row["semseg"], row["instseg"], width, height))
        image, nir, depth, road, boxes = letterbox_all(
            image, nir, depth, road, boxes, self.size
        )
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            nir = np.ascontiguousarray(nir[:, ::-1])
            depth = np.ascontiguousarray(depth[:, ::-1])
            road = np.ascontiguousarray(road[:, ::-1])
            boxes = [(cls, 1 - cx, cy, bw, bh) for cls, cx, cy, bw, bh in boxes]
        if self.augment:
            gain, bias = random.uniform(0.75, 1.25), random.uniform(-18, 18)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        valid_depth = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        return {
            "img": torch.from_numpy(image).permute(2, 0, 1).float() / 255,
            "nir": torch.from_numpy(nir).unsqueeze(0).float() / 255,
            "depth": torch.from_numpy(depth),
            "valid_depth": torch.from_numpy(valid_depth),
            "road": torch.from_numpy(road.astype(np.int64)),
            "boxes": boxes,
            "date": row["date"],
            "name": row["name"],
            "nir_name": nir_row["name"],
        }

class DepthLoss(nn.Module):
    def forward(self, prediction, target, valid):
        prediction, target = prediction[valid].clamp_min(1e-6), target[valid].clamp_min(1e-6)
        if prediction.numel() < 10:
            return prediction.sum() * 0
        difference = torch.log(prediction) - torch.log(target)
        silog = torch.sqrt(
            ((difference.square()).mean() - 0.85 * difference.mean().square()).clamp_min(0) + 1e-6
        )
        # Explicit log-L1 anchors metric scale, which pure SiLog weakly constrains.
        return 0.85 * silog + 0.15 * difference.abs().mean()


class RoadLoss(nn.Module):
    def forward(self, logits, target):
        valid = target != 255
        if not valid.any():
            return logits.sum() * 0
        labels = target[valid].float()
        values = logits[valid]
        positive = labels.sum().clamp_min(1)
        negative = (1 - labels).sum().clamp_min(1)
        pos_weight = (negative / positive).clamp(1, 10).detach()
        bce = F.binary_cross_entropy_with_logits(values, labels, pos_weight=pos_weight)
        probability = values.sigmoid()
        dice = 1 - (2 * (probability * labels).sum() + 1) / (probability.sum() + labels.sum() + 1)
        return bce + dice

def update_depth(totals, prediction, target, valid):
    prediction, target = prediction[valid].clamp_min(1e-4), target[valid].clamp_min(1e-4)
    difference = prediction - target
    totals["pixels"] += prediction.numel()
    totals["abs_rel_sum"] += (difference.abs() / target).sum().item()
    totals["square_sum"] += difference.square().sum().item()
    totals["delta1_sum"] += (torch.maximum(prediction / target, target / prediction) < 1.25).sum().item()


def update_road(totals, logits, target):
    valid = target != 255
    prediction, target = logits[valid].sigmoid() >= 0.5, target[valid] == 1
    totals["road_tp"] += (prediction & target).sum().item()
    totals["road_fp"] += (prediction & ~target).sum().item()
    totals["road_fn"] += (~prediction & target).sum().item()


def run_epoch(model, loader, det_criterion, depth_fn, road_fn, args, device, optimizer, scaler):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    for batch in loader:
        for key in ("img", "depth", "valid_depth", "road", "batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            "cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"
        ):
            detection, depth, road = model(batch["img"])
            det_sum, det_items = det_criterion(detection, batch)
            det_loss = det_sum.sum() / batch["img"].shape[0]
            depth_loss = depth_fn(depth, batch["depth"], batch["valid_depth"])
            road_loss = road_fn(road, batch["road"])
            loss = det_loss + args.depth_weight * depth_loss + args.road_weight * road_loss
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer); scaler.update()

        count = batch["img"].shape[0]
        totals["samples"] += count

        for key, value in (("loss", loss), ("det_loss", det_loss), ("depth_loss", depth_loss), ("road_loss", road_loss)):
            totals[key] += value.item() * count
        totals["box_loss"] += det_items[0].item() * count
        totals["cls_loss"] += det_items[1].item() * count
        totals["dfl_loss"] += det_items[2].item() * count

        update_depth(totals, depth.detach(), batch["depth"], batch["valid_depth"])
        update_road(totals, road.detach(), batch["road"])

    samples, pixels = max(1, totals["samples"]), max(1, totals["pixels"])
    tp, fp, fn = totals["road_tp"], totals["road_fp"], totals["road_fn"]
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)

    return {
        "loss": totals["loss"] / samples,
        "det_loss": totals["det_loss"] / samples,
        "depth_loss": totals["depth_loss"] / samples,
        "road_loss": totals["road_loss"] / samples,
        "box_loss": totals["box_loss"] / samples,
        "cls_loss": totals["cls_loss"] / samples,
        "dfl_loss": totals["dfl_loss"] / samples,
        "abs_rel": totals["abs_rel_sum"] / pixels,
        "rmse": math.sqrt(totals["square_sum"] / pixels),
        "delta1": totals["delta1_sum"] / pixels,
        "road_iou": tp / max(1, tp + fp + fn),
        "road_precision": precision,
        "road_recall": recall,
        "road_f1": 2 * precision * recall / max(1e-12, precision + recall),
    }

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, args):
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "best": best, "args": vars(args),
    }, path)

def save_detector(path, model):
    # Ultralytics-compatible detection-only export for existing test visualization.
    detector = deepcopy(model.detector).to("cpu").half().eval()
    torch.save(
        {
            "model": detector,
            "ema": None,
            "optimizer": None,
            "train_args": vars(detector.args),
            "date": datetime.now().isoformat(),
            "version": ultralytics_version,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        },
        path,
    )


def build_four_class_detector(model_path, device):
    """Build a 4-class YOLO11 detector and transfer compatible pretrained weights."""
    pretrained = YOLO(model_path).model
    source_args = getattr(pretrained, "args", {})
    if not isinstance(source_args, dict):
        source_args = vars(source_args)

    if int(pretrained.yaml.get("nc", 0)) == len(DETECTION_NAMES):
        detector = pretrained
    else:
        detector = DetectionModel(
            deepcopy(pretrained.yaml),
            ch=pretrained.yaml.get("channels", 3),
            nc=len(DETECTION_NAMES),
            verbose=True,
        )
        detector.load(pretrained, verbose=True)

    detector.names = DETECTION_NAMES.copy()
    detector.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **source_args})
    return detector.to(device)

def main():
    args = parse_args()
    image_size = resolve_image_size(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is unavailable; use --device cpu fpr a CPU smoke test")
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device.split(',')[0]}")

    detector = build_four_class_detector(args.model, device)

    model = YoloDepthDetRoad(detector, args.min_depth).to(device)
    if args.mtl_init and Path(args.mtl_init).exists() and not args.resume:
        old = torch.load(args.mtl_init, map_location='cpu', weights_only=False)["model"]
        detector_state = {key[len("detector."):]: value for key, value in old.items() if key.startswith("detector.")}
        result = model.detector.load_state_dict(detector_state, strict=False)
        print(f"Transferred 2-task detector: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    model.to(device)

    det_criterion = model.detector.init_criterion()
    optimizer = AdamW([
        {"params": model.detector.parameters(), "lr": args.lr},
        {"params": model.dense_head.parameters(), "lr": args.lr * args.dense_lr_mult},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start, best = 1, {"joint": float("inf"), "depth": float("inf"), "road": -1.0}

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]

    train_set = ThreeTaskDataset(args.dataset_root, "train", image_size, args.min_depth, args.max_depth, True)
    val_set = ThreeTaskDataset(args.dataset_root, "test", image_size, args.min_depth, args.max_depth, False)
    date_counts = defaultdict(int)

    for row in train_set.rows: date_counts[row["date"]] += 1
    weights = [1 / date_counts[row["date"]] for row in train_set.rows]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                              pin_memory=device.type == "cuda", collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", collate_fn=collate)

    print(
        f"train={len(train_set)} val={len(val_set)} dates={dict(date_counts)} "
        f"input={image_size[0]}x{image_size[1]} device={device}"
    )
    fields = ["epoch", "seconds"] + [f"{split}_{key}" for split in ("train", "val") for key in (
        "loss", "det_loss", "depth_loss", "road_loss", "box_loss", "cls_loss", "dfl_loss",
        "abs_rel", "rmse", "delta1", "road_iou", "road_precision", "road_recall", "road_f1")]
    history = output_dir / "history.csv"

    depth_fn, road_fn = DepthLoss(), RoadLoss()

    for epoch in range(start, args.epochs + 1):
        begin = time.time()

        train = run_epoch(
            model, train_loader, det_criterion, depth_fn, road_fn,
            args, device, optimizer, scaler,
        )
        val = run_epoch(model, val_loader, det_criterion, depth_fn, road_fn, args, device, None, scaler)

        scheduler.step()

        row = {"epoch": epoch, "seconds": time.time() - begin}
        row.update({f"train_{key}": value for key, value in train.items()})
        row.update({f"val_{key}": value for key, value in val.items()})
        write_header = not history.exists() or (start == 1 and epoch == 1)
        with history.open("w" if write_header else "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header: writer.writeheader()
            writer.writerow(row)
        joint = val["det_loss"] + args.depth_weight * val["depth_loss"] + args.road_weight * val["road_loss"]
        improved = {"joint": joint < best["joint"], "depth": val["abs_rel"] < best["depth"], "road": val["road_iou"] > best["road"]}
        if improved["joint"]: best["joint"] = joint
        if improved["depth"]: best["depth"] = val["abs_rel"]
        if improved["road"]: best["road"] = val["road_iou"]

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best, args)
        save_detector(output_dir / "last_detector.pt", model)
        for key in ("joint", "depth", "road"):
            if improved[key]: save_checkpoint(output_dir / f"best_{key}.pt", model, optimizer, scheduler, scaler, epoch, best, args)

        print(f"epoch {epoch:03d}/{args.epochs} train={train['loss']:.4f} | val={val['loss']:.4f} "
              f"det={val['det_loss']:.4f} depth={val['depth_loss']:.4f} AbsRel={val['abs_rel']:.4f} "
              f"RMSE={val['rmse']:.3f} d1={val['delta1']:.4f} roadIoU={val['road_iou']:.4f} roadF1={val['road_f1']:.4f}")

if __name__ == "__main__":
    main()
