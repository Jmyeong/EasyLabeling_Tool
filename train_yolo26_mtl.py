#!/usr/bin/env python3
"""Train a joint YOLO26 model for detection, metric depth, and Road segmentation.

The detector is a native YOLO26 end-to-end model. The revised dense decoder
uses P1-P5 detector features, task-specific refinement, and bounded log-depth.
By default RGB and NIR use independent P1 stems and learned spatial/channel
gates before entering the shared YOLO26 graph. Depth-only and legacy modes remain
available for controlled comparisons.
The original YOLO workspace and copied baseline checkpoints are not modified.
"""

import argparse
from dated_dataset import add_dataset_arguments, dataset_options
import csv
from copy import deepcopy
from datetime import datetime
import json
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import sys
import time
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace

# Reuse the exact data preprocessing and dense-task losses used by YOLO11.
# This keeps the YOLO11/YOLO26 comparison focused on the detector backbone/head.
from train import (
    RoadLoss,
    ThreeTaskDataset,
    collate,
    normalize_image_size,
    resolve_dataset_manifest_root,
    resolve_image_size,
    update_depth,
    update_road,
)
from visualize_yolo_test import match_detections

CLASS_NAMES = {0: "Golfcart", 1: "Person", 2: "Tree", 3: "Undef_obj"}
MIN_ULTRALYTICS_VERSION = (8, 4, 0)
PROJECT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Arguments and environment
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26 detection + metric depth + Road segmentation training"
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "checkpoints/yolo26s.pt"),
        help="COCO-pretrained YOLO26 detection checkpoint",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "runs" / "yolo26s_depth_only_nir_fusion_mtl"),
    )
    parser.add_argument(
        "--input-mode",
        choices=(
            "depth_nir_gated", "depth_nir", "scalar", "gated", "rgbn", "hsvnet"
        ),
        default="gated",
        help=(
            "depth_nir_gated uses normalized, quality-gated NIR only for depth; "
            "depth_nir is the legacy scalar depth-only fusion; "
            "scalar fuses shared P1 features with one residual weight; "
            "gated uses spatial/channel gates; "
            "rgbn concatenates four raw channels; hsvnet uses the legacy front-end"
        ),
    )
    parser.add_argument(
        "--fusion-repo",
        default="external/Pixel_aligned_RGB_NIR_Stereo",
        help="Repository containing net/image_fusion.py from Pixel-aligned RGB-NIR Stereo",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        default=(
            "external/Pixel_aligned_RGB_NIR_Stereo/"
            "weights/model_image_fusion.pth"
        ),
        help="Pretrained HSVNet image-fusion checkpoint",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--height", type=int, default=352, help="Network input height")
    parser.add_argument("--width", type=int, default=640, help="Network input width")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Backward-compatible alias for --width (height remains --height)",
    )
    parser.add_argument("--device", default="3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dense-lr-mult", type=float, default=10.0)
    parser.add_argument("--fusion-lr-mult", type=float, default=1.0)
    parser.add_argument(
        "--feature-fusion-lr-mult",
        "--gated-fusion-lr-mult",
        dest="feature_fusion_lr_mult",
        type=float,
        default=10.0,
        help="Learning-rate multiplier for the new NIR stem and fusion parameters",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth-weight", type=float, default=0.50)
    parser.add_argument("--road-weight", type=float, default=0.50)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=35.0)
    parser.add_argument(
        "--depth-init",
        type=float,
        default=8.0,
        help="Initial metric depth for the bounded log-depth output",
    )
    parser.add_argument("--depth-silog-weight", type=float, default=0.85)
    parser.add_argument("--depth-log-l1-weight", type=float, default=0.15)
    parser.add_argument("--depth-gradient-weight", type=float, default=0.15)
    parser.add_argument("--depth-smoothness-weight", type=float, default=0.02)
    parser.add_argument(
        "--head-warmup-epochs",
        type=int,
        default=5,
        help="Train fusion and dense heads, with YOLO frozen, for the first N epochs",
    )
    parser.add_argument(
        "--nir-dropout",
        type=float,
        default=0.20,
        help="Training probability of replacing a sample's NIR with zeros in gated depth mode",
    )
    parser.add_argument(
        "--nir-mismatch",
        type=float,
        default=0.20,
        help="Training probability of pairing NIR from another sample of the same date",
    )

    parser.add_argument(
        "--resume",
        default="",
        help="Resume a joint checkpoint such as last.pt",
    )
    parser.add_argument(
        "--val-conf", type=float, default=0.20,
        help="Detection confidence threshold used during validation",
    )
    parser.add_argument(
        "--val-eval-iou", type=float, default=0.30,
        help="Class-aware GT matching IoU used during validation",
    )
    parser.add_argument(
        "--val-max-det", type=int, default=300,
        help="Maximum YOLO26 detections per validation image",
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def version_tuple(version_text):
    numbers = re.findall(r"\d+", version_text.split("+")[0])
    return tuple(int(value) for value in numbers[:3])


def validate_environment(args):
    if version_tuple(ultralytics_version) < MIN_ULTRALYTICS_VERSION:
        raise RuntimeError(
            "YOLO26 requires ultralytics>=8.4.0, but this environment has "
            f"{ultralytics_version}. Use a separate YOLO26 environment."
        )
    image_size = resolve_image_size(args)
    if any(dimension % 32 for dimension in image_size):
        raise ValueError(
            f"Input height and width must be divisible by 32, got {image_size}"
        )
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a smoke test")
    if not args.min_depth < args.depth_init < args.max_depth:
        raise ValueError(
            "--depth-init must be strictly between --min-depth and --max-depth"
        )
    if args.head_warmup_epochs < 0:
        raise ValueError("--head-warmup-epochs cannot be negative")
    for name in ("nir_dropout", "nir_mismatch"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.input_mode == "hsvnet":
        for label, path in (
            ("fusion repository", Path(args.fusion_repo)),
            ("fusion checkpoint", Path(args.fusion_checkpoint)),
        ):
            if str(path) not in ("", ".") and not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")


def select_device(device_text):
    if device_text == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{str(device_text).split(',')[0]}")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pixel_aligned_image_fusion(args, device):
    """Load the pretrained HSV channel-attention fusion model from its source repo."""
    repository = str(Path(args.fusion_repo).expanduser().resolve())
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from net.image_fusion import HSVNet

    fusion = HSVNet()
    checkpoint = torch.load(
        Path(args.fusion_checkpoint).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    fusion.load_state_dict(state_dict, strict=True)
    print(f"RGB-NIR image fusion: HSVNet ({args.fusion_checkpoint})")
    return fusion.to(device)


# -----------------------------------------------------------------------------
# YOLO26 detector and shared multi-scale features
# -----------------------------------------------------------------------------


def make_detector_args(source_args, args):
    """Create the hyperparameter namespace required by the YOLO26 criterion."""
    if not isinstance(source_args, dict):
        source_args = vars(source_args)
    values = {**DEFAULT_CFG_DICT, **source_args}
    values.update(
        {
            "epochs": args.epochs,
            "imgsz": list(resolve_image_size(args)),
            "data": str(resolve_dataset_manifest_root(args.dataset_root) / "data.yaml"),
        }
    )
    return IterableSimpleNamespace(**values)


def build_yolo26_detector(args, device):
    """Build a four-class YOLO26 graph and transfer compatible COCO weights."""
    pretrained = YOLO(args.model).model
    yaml = deepcopy(pretrained.yaml)
    if not yaml.get("end2end", False) or int(yaml.get("reg_max", 0)) != 1:
        raise ValueError(
            f"{args.model} is not a native YOLO26 detection model "
            "(expected end2end=True and reg_max=1)"
        )

    input_channels = 4 if args.input_mode == "rgbn" else 3
    detector = DetectionModel(
        yaml, ch=input_channels, nc=len(CLASS_NAMES), verbose=True
    )
    detector.names = CLASS_NAMES.copy()
    detector.args = make_detector_args(getattr(pretrained, "args", {}), args)
    detector.load(pretrained, verbose=True)
    if input_channels == 4:
        # Ultralytics transfers the three compatible RGB slices automatically.
        # Initialize the extra NIR slice from their mean instead of leaving it
        # random, preserving the scale of the pretrained first-layer response.
        first_weight = detector.model[0].conv.weight
        with torch.no_grad():
            first_weight[:, 3:4].copy_(first_weight[:, :3].mean(dim=1, keepdim=True))
        print("RGBN early fusion: first convolution initialized RGB + mean(RGB) NIR")
    detector = detector.to(device)

    head = detector.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("The constructed detector lost YOLO26 end-to-end mode")
    print(
        "YOLO26 detector: "
        f"classes={head.nc}, reg_max={head.reg_max}, end2end={head.end2end}"
    )
    return detector


def build_gated_feature_fusion(detector, device):
    fusion = GatedRGBNStem(detector.model[0])
    print(
        "RGB-NIR gated fusion: separate P1 stems, channel+spatial gate, "
        "initial gate=sigmoid(-4)"
    )
    return fusion.to(device)


def build_scalar_feature_fusion(detector, device):
    fusion = ScalarRGBNStem(detector.model[0])
    print(
        "RGB-NIR scalar fusion: separate P1 stems, "
        "fused=RGB+alpha*NIR, initial alpha=0"
    )
    return fusion.to(device)


def build_depth_nir_encoder(detector, device, quality_gated=False):
    encoder = DepthNIRPyramid(detector.model[0])
    description = (
        "percentile-normalized NIR pyramid with learned quality/spatial gates"
        if quality_gated
        else "legacy NIR pyramid with scalar residual weights"
    )
    print(f"Depth-only NIR fusion: RGB-only detector/Road, {description}")
    return encoder.to(device)


def execute_graph(detector, image, capture_indices=(), injected_outputs=None):
    """Execute an Ultralytics graph while retaining requested intermediate maps."""
    saved_outputs = []
    captured = {}
    x = image
    capture_indices = set(capture_indices)
    injected_outputs = injected_outputs or {}

    for module in detector.model:
        if module.i in injected_outputs:
            x = injected_outputs[module.i]
        else:
            if module.f != -1:
                if isinstance(module.f, int):
                    x = saved_outputs[module.f]
                else:
                    x = [x if index == -1 else saved_outputs[index] for index in module.f]
            x = module(x)
        saved_outputs.append(x if module.i in detector.save else None)
        if module.i in capture_indices:
            captured[module.i] = x
    return x, captured


@torch.no_grad()
def discover_dense_features(detector, image_size, device, include_p1=True):
    """Find P1/P2 plus the three feature maps feeding the Detect head."""
    detect_indices = [int(index) for index in detector.model[-1].f]
    if len(detect_indices) != 3:
        raise ValueError(f"Expected three detection scales, received {detect_indices}")

    # Keep every intermediate output only during this one discovery pass.
    outputs = []
    image_height, image_width = normalize_image_size(image_size)
    input_channels = int(detector.model[0].conv.in_channels)
    x = torch.zeros(1, input_channels, image_height, image_width, device=device)
    was_training = detector.training
    detector.eval()
    for module in detector.model:
        if module.f != -1:
            if isinstance(module.f, int):
                x = outputs[module.f]
            else:
                x = [x if index == -1 else outputs[index] for index in module.f]
        x = module(x)
        outputs.append(x)
    detector.train(was_training)

    p3 = outputs[detect_indices[0]]
    expected_p2_size = (p3.shape[-2] * 2, p3.shape[-1] * 2)
    p2_candidates = [
        index
        for index, output in enumerate(outputs[: detect_indices[0]])
        if isinstance(output, torch.Tensor) and tuple(output.shape[-2:]) == expected_p2_size
    ]
    if not p2_candidates:
        raise RuntimeError("Could not discover a P2/4 feature map in the YOLO26 graph")

    p2_index = p2_candidates[-1]
    if include_p1:
        p2 = outputs[p2_index]
        expected_p1_size = (p2.shape[-2] * 2, p2.shape[-1] * 2)
        p1_candidates = [
            index
            for index, output in enumerate(outputs[:p2_index])
            if isinstance(output, torch.Tensor)
            and tuple(output.shape[-2:]) == expected_p1_size
        ]
        if not p1_candidates:
            raise RuntimeError("Could not discover a P1/2 feature map in the YOLO26 graph")
        feature_indices = (p1_candidates[-1], p2_index, *detect_indices)
    else:
        # Legacy baseline checkpoints used P2-P5 only.
        feature_indices = (p2_index, *detect_indices)
    feature_channels = tuple(int(outputs[index].shape[1]) for index in feature_indices)
    feature_sizes = tuple(tuple(outputs[index].shape[-2:]) for index in feature_indices)
    print(
        f"Dense features: indices={feature_indices}, "
        f"channels={feature_channels}, sizes={feature_sizes}"
    )
    return feature_indices, feature_channels


# -----------------------------------------------------------------------------
# Depth and Road decoder
# -----------------------------------------------------------------------------


def group_count(channels):
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(channels, channels), ConvBlock(channels, channels)
        )

    def forward(self, inputs):
        return inputs + self.blocks(inputs)


class GatedRGBNStem(nn.Module):
    """Fuse independent RGB/NIR P1 features with channel and spatial gates.

    The RGB feature is produced by the detector's pretrained first layer. This
    module owns only the NIR stem and gate branches. Its residual gate starts
    near zero, so the initial network remains close to the RGB-only detector.
    """

    def __init__(self, rgb_stem, initial_gate=-4.0):
        super().__init__()
        rgb_conv = rgb_stem.conv
        out_channels = int(rgb_conv.out_channels)
        self.nir_stem = deepcopy(rgb_stem)
        self.nir_stem.conv = nn.Conv2d(
            1,
            out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=rgb_conv.bias is not None,
            padding_mode=rgb_conv.padding_mode,
        ).to(device=rgb_conv.weight.device, dtype=rgb_conv.weight.dtype)
        with torch.no_grad():
            self.nir_stem.conv.weight.copy_(
                rgb_conv.weight.mean(dim=1, keepdim=True)
            )
            if rgb_conv.bias is not None:
                self.nir_stem.conv.bias.copy_(rgb_conv.bias)

        hidden_channels = max(8, out_channels // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )
        self.spatial_gate = nn.Conv2d(out_channels * 2, 1, 3, padding=1)
        self.gate_bias = nn.Parameter(torch.tensor(float(initial_gate)))

        # All learned gate residuals start at zero; sigmoid(-4) ~= 0.018.
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

    def forward(self, rgb_feature, nir):
        nir_feature = self.nir_stem(nir)
        if nir_feature.shape != rgb_feature.shape:
            raise RuntimeError(
                f"RGB/NIR stem mismatch: {tuple(rgb_feature.shape)} vs "
                f"{tuple(nir_feature.shape)}"
            )
        combined = torch.cat([rgb_feature, nir_feature], dim=1)
        gate = torch.sigmoid(
            self.channel_gate(combined) + self.spatial_gate(combined) + self.gate_bias
        )
        return rgb_feature + gate * nir_feature, gate


class ScalarRGBNStem(nn.Module):
    """Lightweight dual stem with one learnable NIR residual coefficient."""

    def __init__(self, rgb_stem):
        super().__init__()
        rgb_conv = rgb_stem.conv
        out_channels = int(rgb_conv.out_channels)
        self.nir_stem = deepcopy(rgb_stem)
        self.nir_stem.conv = nn.Conv2d(
            1,
            out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=rgb_conv.bias is not None,
            padding_mode=rgb_conv.padding_mode,
        ).to(device=rgb_conv.weight.device, dtype=rgb_conv.weight.dtype)
        with torch.no_grad():
            self.nir_stem.conv.weight.copy_(
                rgb_conv.weight.mean(dim=1, keepdim=True)
            )
            if rgb_conv.bias is not None:
                self.nir_stem.conv.bias.copy_(rgb_conv.bias)

        # Exactly RGB-only at initialization. Alpha receives a gradient on the
        # first step; the NIR stem starts learning once alpha moves away from 0.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, rgb_feature, nir):
        nir_feature = self.nir_stem(nir)
        if nir_feature.shape != rgb_feature.shape:
            raise RuntimeError(
                f"RGB/NIR stem mismatch: {tuple(rgb_feature.shape)} vs "
                f"{tuple(nir_feature.shape)}"
            )
        return rgb_feature + self.alpha * nir_feature, self.alpha


class DepthNIRPyramid(nn.Module):
    """Shallow NIR encoder producing P1/2 and P2/4 depth features."""

    def __init__(self, rgb_stem):
        super().__init__()
        rgb_conv = rgb_stem.conv
        p1_channels = int(rgb_conv.out_channels)
        self.p1 = deepcopy(rgb_stem)
        self.p1.conv = nn.Conv2d(
            1,
            p1_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=rgb_conv.bias is not None,
            padding_mode=rgb_conv.padding_mode,
        ).to(device=rgb_conv.weight.device, dtype=rgb_conv.weight.dtype)
        with torch.no_grad():
            self.p1.conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
            if rgb_conv.bias is not None:
                self.p1.conv.bias.copy_(rgb_conv.bias)

        self.p2 = nn.Sequential(
            nn.Conv2d(p1_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(64), 64),
            nn.SiLU(inplace=True),
        )

    def forward(self, nir):
        p1 = self.p1(nir)
        return p1, self.p2(p1)


class LegacyDenseDepthRoadHead26(nn.Module):
    """Original P2-P5 decoder retained for baseline checkpoint evaluation."""

    def __init__(self, feature_channels, min_depth):
        super().__init__()
        p2_channels, p3_channels, p4_channels, p5_channels = feature_channels
        self.min_depth = float(min_depth)

        self.p2 = ConvBlock(p2_channels, 48)
        self.p3 = ConvBlock(p3_channels, 64)
        self.p4 = ConvBlock(p4_channels, 64)
        self.p5 = ConvBlock(p5_channels, 64)
        self.context = ConvBlock(192, 96)
        self.refine = nn.Sequential(ConvBlock(144, 96), ConvBlock(96, 64))
        self.depth_output = nn.Sequential(ConvBlock(64, 32), nn.Conv2d(32, 1, 1))
        self.road_output = nn.Sequential(ConvBlock(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, features, output_size):
        p2, p3, p4, p5 = features
        p3_size = p3.shape[-2:]
        context = self.context(
            torch.cat(
                [
                    self.p3(p3),
                    F.interpolate(self.p4(p4), p3_size, mode="bilinear", align_corners=False),
                    F.interpolate(self.p5(p5), p3_size, mode="bilinear", align_corners=False),
                ],
                dim=1,
            )
        )
        context = F.interpolate(context, p2.shape[-2:], mode="bilinear", align_corners=False)
        dense = self.refine(torch.cat([self.p2(p2), context], dim=1))

        depth = F.interpolate(
            self.depth_output(dense), output_size, mode="bilinear", align_corners=False
        )[:, 0]
        road = F.interpolate(
            self.road_output(dense), output_size, mode="bilinear", align_corners=False
        )[:, 0]
        return F.softplus(depth) + self.min_depth, road


class DenseDepthRoadHead26(nn.Module):
    """P1-P5 top-down decoder with independent depth and Road refinement."""

    def __init__(
        self,
        feature_channels,
        min_depth,
        max_depth,
        initial_depth,
        use_depth_nir=False,
        quality_gated_nir=False,
    ):
        super().__init__()
        p1_channels, p2_channels, p3_channels, p4_channels, p5_channels = feature_channels
        self.register_buffer("log_min_depth", torch.tensor(math.log(float(min_depth))))
        self.register_buffer(
            "log_depth_range",
            torch.tensor(math.log(float(max_depth)) - math.log(float(min_depth))),
        )

        self.p1 = ConvBlock(p1_channels, 32)
        self.p2 = ConvBlock(p2_channels, 64)
        self.p3 = ConvBlock(p3_channels, 96)
        self.p4 = ConvBlock(p4_channels, 96)
        self.p5 = ConvBlock(p5_channels, 96)
        self.fuse_p4 = nn.Sequential(ConvBlock(192, 96), ResidualBlock(96))
        self.fuse_p3 = nn.Sequential(ConvBlock(192, 96), ResidualBlock(96))
        self.fuse_p2 = nn.Sequential(ConvBlock(160, 64), ResidualBlock(64))

        # Multi-scale context is shared, but task-specific spatial refinement is not.
        self.depth_p2 = nn.Sequential(ResidualBlock(64), ConvBlock(64, 64))
        self.road_p2 = nn.Sequential(ResidualBlock(64), ConvBlock(64, 64))
        self.depth_p1 = nn.Sequential(
            ConvBlock(96, 64), ResidualBlock(64), ConvBlock(64, 32)
        )
        self.road_p1 = nn.Sequential(
            ConvBlock(96, 64), ResidualBlock(64), ConvBlock(64, 32)
        )
        self.depth_output = nn.Conv2d(32, 1, 1)
        self.road_output = nn.Conv2d(32, 1, 1)
        self.use_depth_nir = bool(use_depth_nir)
        self.quality_gated_nir = bool(quality_gated_nir)
        if self.use_depth_nir:
            # P1 and P2 residual weights start at zero, exactly reproducing the
            # RGB depth path before learning whether NIR is useful.
            self.depth_nir_alpha = nn.Parameter(torch.zeros(2))
        if self.quality_gated_nir:
            quality_hidden = 16
            self.depth_nir_quality_gate = nn.Sequential(
                nn.Linear(8, quality_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(quality_hidden, 2),
            )
            self.depth_nir_spatial_gate_p1 = nn.Conv2d(64, 1, 3, padding=1)
            self.depth_nir_spatial_gate_p2 = nn.Conv2d(128, 1, 3, padding=1)
            self.depth_nir_gate_bias = nn.Parameter(torch.full((2,), -1.0))
            # Alpha starts at zero, so the new model is exactly RGB-only before
            # training. Gates become trainable as soon as alpha leaves zero.
            nn.init.zeros_(self.depth_nir_quality_gate[-1].weight)
            nn.init.zeros_(self.depth_nir_quality_gate[-1].bias)
            nn.init.zeros_(self.depth_nir_spatial_gate_p1.weight)
            nn.init.zeros_(self.depth_nir_spatial_gate_p1.bias)
            nn.init.zeros_(self.depth_nir_spatial_gate_p2.weight)
            nn.init.zeros_(self.depth_nir_spatial_gate_p2.bias)
        self.last_depth_nir_gate_mean = None

        normalized = (
            math.log(float(initial_depth)) - math.log(float(min_depth))
        ) / (math.log(float(max_depth)) - math.log(float(min_depth)))
        initial_bias = math.log(normalized / (1.0 - normalized))
        nn.init.normal_(self.depth_output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.depth_output.bias, initial_bias)

    @staticmethod
    def upsample(inputs, reference):
        return F.interpolate(
            inputs, reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self, features, output_size, nir_features=None, modality_quality=None
    ):
        p1, p2, p3, p4, p5 = features
        top_p5 = self.p5(p5)
        top_p4 = self.fuse_p4(
            torch.cat([self.p4(p4), self.upsample(top_p5, p4)], dim=1)
        )
        top_p3 = self.fuse_p3(
            torch.cat([self.p3(p3), self.upsample(top_p4, p3)], dim=1)
        )
        top_p2 = self.fuse_p2(
            torch.cat([self.p2(p2), self.upsample(top_p3, p2)], dim=1)
        )

        rgb_p1 = self.p1(p1)
        depth_top_p2 = top_p2
        depth_rgb_p1 = rgb_p1
        if self.use_depth_nir:
            if nir_features is None:
                raise ValueError("Depth-only NIR head requires P1/P2 NIR features")
            nir_p1, nir_p2 = nir_features
            if nir_p1.shape != rgb_p1.shape or nir_p2.shape != top_p2.shape:
                raise RuntimeError(
                    "Depth NIR feature mismatch: "
                    f"nir_p1={tuple(nir_p1.shape)} rgb_p1={tuple(rgb_p1.shape)} "
                    f"nir_p2={tuple(nir_p2.shape)} rgb_p2={tuple(top_p2.shape)}"
                )
            if self.quality_gated_nir:
                if modality_quality is None or modality_quality.shape[-1] != 8:
                    raise ValueError(
                        "Quality-gated NIR fusion requires [B,8] modality statistics"
                    )
                quality_logits = self.depth_nir_quality_gate(modality_quality)
                gate_p1 = torch.sigmoid(
                    self.depth_nir_spatial_gate_p1(
                        torch.cat([rgb_p1, nir_p1], dim=1)
                    )
                    + quality_logits[:, 0, None, None, None]
                    + self.depth_nir_gate_bias[0]
                )
                gate_p2 = torch.sigmoid(
                    self.depth_nir_spatial_gate_p2(
                        torch.cat([top_p2, nir_p2], dim=1)
                    )
                    + quality_logits[:, 1, None, None, None]
                    + self.depth_nir_gate_bias[1]
                )
                depth_rgb_p1 = rgb_p1 + self.depth_nir_alpha[0] * gate_p1 * nir_p1
                depth_top_p2 = top_p2 + self.depth_nir_alpha[1] * gate_p2 * nir_p2
                self.last_depth_nir_gate_mean = torch.stack(
                    [
                        gate_p1.mean(dim=(1, 2, 3)),
                        gate_p2.mean(dim=(1, 2, 3)),
                    ],
                    dim=1,
                ).detach()
            else:
                depth_rgb_p1 = rgb_p1 + self.depth_nir_alpha[0] * nir_p1
                depth_top_p2 = top_p2 + self.depth_nir_alpha[1] * nir_p2

        depth_p1 = self.depth_p1(
            torch.cat(
                [depth_rgb_p1, self.upsample(self.depth_p2(depth_top_p2), p1)],
                dim=1,
            )
        )
        road_p1 = self.road_p1(
            torch.cat(
                [rgb_p1, self.upsample(self.road_p2(top_p2), p1)], dim=1
            )
        )
        depth_logit = F.interpolate(
            self.depth_output(depth_p1),
            output_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        road = F.interpolate(
            self.road_output(road_p1),
            output_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        log_depth = self.log_min_depth + depth_logit.sigmoid() * self.log_depth_range
        return log_depth.exp(), road


class Yolo26DepthDetRoad(nn.Module):
    def __init__(
        self,
        detector,
        feature_indices,
        feature_channels,
        min_depth,
        max_depth=35.0,
        initial_depth=8.0,
        image_fusion=None,
        feature_fusion=None,
        nir_encoder=None,
        input_mode="rgbn",
    ):
        super().__init__()
        self.detector = detector
        self.feature_indices = tuple(feature_indices)
        self.input_mode = str(input_mode)
        if self.input_mode not in (
            "depth_nir_gated", "depth_nir", "scalar", "gated", "rgbn", "hsvnet"
        ):
            raise ValueError(f"Unsupported input mode: {self.input_mode}")
        if self.input_mode == "hsvnet" and image_fusion is None:
            raise ValueError("HSVNet input mode requires an image-fusion model")
        if self.input_mode in ("scalar", "gated") and feature_fusion is None:
            raise ValueError(
                f"{self.input_mode} input mode requires a P1 feature-fusion module"
            )
        if self.input_mode in ("depth_nir", "depth_nir_gated") and nir_encoder is None:
            raise ValueError("Depth-only NIR modes require a shallow NIR encoder")
        self.image_fusion = image_fusion
        self.feature_fusion = feature_fusion
        self.nir_encoder = nir_encoder
        self.last_gate_mean = None
        self.last_depth_nir_alpha = None
        self.last_depth_nir_gate_mean = None
        if len(feature_indices) == 4:
            if self.input_mode in ("depth_nir", "depth_nir_gated"):
                raise ValueError("Depth-only NIR modes require the revised P1-P5 decoder")
            self.architecture_version = "legacy_p2_p5"
            self.dense_head = LegacyDenseDepthRoadHead26(feature_channels, min_depth)
        elif len(feature_indices) == 5:
            self.architecture_version = (
                "v3_p1_p5_depth_nir_quality_gated"
                if self.input_mode == "depth_nir_gated"
                else "v2_p1_p5_depth_only_nir"
                if self.input_mode == "depth_nir"
                else "v1_p1_p5_separate"
            )
            self.dense_head = DenseDepthRoadHead26(
                feature_channels,
                min_depth,
                max_depth,
                initial_depth,
                use_depth_nir=self.input_mode in ("depth_nir", "depth_nir_gated"),
                quality_gated_nir=self.input_mode == "depth_nir_gated",
            )
        else:
            raise ValueError(f"Expected four or five dense features, got {feature_indices}")

    def fuse_inputs(self, image, nir=None):
        if self.input_mode == "rgbn" and nir is None and image.shape[1] == 4:
            # Ultralytics profiling utilities infer four input channels from the
            # detector's first convolution and invoke the wrapper directly.
            return image
        if nir is None:
            # Profilers in Ultralytics invoke the model with a single tensor.
            # A neutral zero-NIR image keeps that interface usable.
            nir = image.new_zeros((image.shape[0], 1, *image.shape[-2:]))
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(image.shape)}")
        if nir.ndim != 4 or nir.shape[1] != 1 or nir.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                f"Expected NIR [B,1,H,W] aligned with RGB, got {tuple(nir.shape)} "
                f"for RGB {tuple(image.shape)}"
            )

        if self.input_mode == "rgbn":
            return torch.cat([image, nir], dim=1)

        # The original attention block uses batch statistics even in eval mode
        # and cannot process B=1 after global pooling. Duplicate only that edge
        # case, then discard the duplicate result.
        single = image.shape[0] == 1
        fusion_rgb = image.repeat(2, 1, 1, 1) if single else image
        fusion_nir = nir.repeat(2, 1, 1, 1) if single else nir
        fused = self.image_fusion(fusion_rgb * 255.0, fusion_nir * 255.0) / 255.0
        if single:
            fused = fused[:1]
        return fused.clamp(0.0, 1.0)

    @staticmethod
    def normalize_nir_percentile(nir):
        """Normalize useful NIR contrast while leaving flat/clipped frames stable."""
        reduced = F.interpolate(nir, size=(44, 80), mode="bilinear", align_corners=False)
        flat = reduced.flatten(2)
        low = torch.quantile(flat, 0.05, dim=2, keepdim=True).unsqueeze(-1)
        high = torch.quantile(flat, 0.95, dim=2, keepdim=True).unsqueeze(-1)
        span = high - low
        normalized = ((nir - low) / span.clamp_min(0.05)).clamp(0.0, 1.0)
        # Near-constant frames contain no robust percentile range. Keeping the
        # raw level lets the quality gate recognize zero or saturated inputs.
        return torch.where(span >= 0.05, normalized, nir)

    @staticmethod
    def modality_statistics(image, nir):
        luminance = (
            0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
        )
        nir_values = nir[:, 0]
        statistics = []
        for values in (luminance, nir_values):
            statistics.extend(
                (
                    values.mean(dim=(1, 2)),
                    values.std(dim=(1, 2), unbiased=False),
                    (values < 0.15).float().mean(dim=(1, 2)),
                    (values > 0.85).float().mean(dim=(1, 2)),
                )
            )
        return torch.stack(statistics, dim=1)

    def forward(self, image, nir=None):
        nir_features = None
        modality_quality = None
        if self.input_mode in ("depth_nir", "depth_nir_gated"):
            if nir is None:
                nir = image.new_zeros((image.shape[0], 1, *image.shape[-2:]))
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(image.shape)}")
            if nir.ndim != 4 or nir.shape[1] != 1 or nir.shape[-2:] != image.shape[-2:]:
                raise ValueError(
                    f"Expected aligned NIR [B,1,H,W], got {tuple(nir.shape)}"
                )
            detection, captured = execute_graph(
                self.detector, image, capture_indices=self.feature_indices
            )
            if self.input_mode == "depth_nir_gated":
                modality_quality = self.modality_statistics(image, nir)
                encoded_nir = self.normalize_nir_percentile(nir)
            else:
                encoded_nir = nir
            nir_features = self.nir_encoder(encoded_nir)
            self.last_depth_nir_alpha = (
                self.dense_head.depth_nir_alpha.detach().clone()
            )
            output_size = image.shape[-2:]
        elif self.input_mode in ("scalar", "gated"):
            if nir is None:
                nir = image.new_zeros((image.shape[0], 1, *image.shape[-2:]))
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(image.shape)}")
            if nir.ndim != 4 or nir.shape[1] != 1 or nir.shape[-2:] != image.shape[-2:]:
                raise ValueError(
                    f"Expected aligned NIR [B,1,H,W], got {tuple(nir.shape)}"
                )
            rgb_p1 = self.detector.model[0](image)
            fused_p1, gate = self.feature_fusion(rgb_p1, nir)
            self.last_gate_mean = gate.detach().mean()
            detection, captured = execute_graph(
                self.detector,
                image,
                capture_indices=self.feature_indices,
                injected_outputs={0: fused_p1},
            )
            output_size = image.shape[-2:]
        else:
            fused_image = self.fuse_inputs(image, nir)
            detection, captured = execute_graph(
                self.detector, fused_image, capture_indices=self.feature_indices
            )
            output_size = fused_image.shape[-2:]
        missing = set(self.feature_indices) - set(captured)
        if missing:
            raise RuntimeError(f"Missing YOLO26 dense features: {sorted(missing)}")
        features = tuple(captured[index] for index in self.feature_indices)
        if self.input_mode in ("depth_nir", "depth_nir_gated"):
            depth, road = self.dense_head(
                features, output_size, nir_features, modality_quality
            )
            self.last_depth_nir_gate_mean = self.dense_head.last_depth_nir_gate_mean
        else:
            depth, road = self.dense_head(features, output_size)
        return detection, depth, road


# -----------------------------------------------------------------------------
# Training and metrics
# -----------------------------------------------------------------------------


class SparseDepthLoss(nn.Module):
    """Metric depth loss for sparse LiDAR supervision and dense regularization."""

    def __init__(self, args):
        super().__init__()
        self.silog_weight = float(args.depth_silog_weight)
        self.log_l1_weight = float(args.depth_log_l1_weight)
        self.gradient_weight = float(args.depth_gradient_weight)
        self.smoothness_weight = float(args.depth_smoothness_weight)

    @staticmethod
    def sparse_gradient(log_prediction, log_target, valid):
        losses = []
        horizontal = valid[:, :, 1:] & valid[:, :, :-1]
        if horizontal.any():
            pred_dx = log_prediction[:, :, 1:] - log_prediction[:, :, :-1]
            target_dx = log_target[:, :, 1:] - log_target[:, :, :-1]
            losses.append((pred_dx - target_dx).abs()[horizontal].mean())
        vertical = valid[:, 1:, :] & valid[:, :-1, :]
        if vertical.any():
            pred_dy = log_prediction[:, 1:, :] - log_prediction[:, :-1, :]
            target_dy = log_target[:, 1:, :] - log_target[:, :-1, :]
            losses.append((pred_dy - target_dy).abs()[vertical].mean())
        if not losses:
            return log_prediction.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def edge_aware_smoothness(prediction, image):
        inverse_depth = prediction.clamp_min(1e-4).reciprocal()
        inverse_depth = inverse_depth / inverse_depth.mean(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-4)
        image_gray = image.mean(dim=1)
        depth_dx = (inverse_depth[:, :, 1:] - inverse_depth[:, :, :-1]).abs()
        depth_dy = (inverse_depth[:, 1:, :] - inverse_depth[:, :-1, :]).abs()
        image_dx = (image_gray[:, :, 1:] - image_gray[:, :, :-1]).abs()
        image_dy = (image_gray[:, 1:, :] - image_gray[:, :-1, :]).abs()
        return (
            (depth_dx * torch.exp(-10.0 * image_dx)).mean()
            + (depth_dy * torch.exp(-10.0 * image_dy)).mean()
        )

    def forward(self, prediction, target, valid, image):
        safe_prediction = prediction.clamp_min(1e-4)
        safe_target = target.clamp_min(1e-4)
        selected_prediction = safe_prediction[valid]
        selected_target = safe_target[valid]
        if selected_prediction.numel() < 10:
            zero = prediction.sum() * 0.0
            return zero, {name: zero for name in ("silog", "log_l1", "gradient", "smoothness")}

        difference = selected_prediction.log() - selected_target.log()
        silog = torch.sqrt(
            (
                difference.square().mean()
                - 0.85 * difference.mean().square()
            ).clamp_min(0.0)
            + 1e-6
        )
        log_l1 = difference.abs().mean()
        gradient = self.sparse_gradient(
            safe_prediction.log(), safe_target.log(), valid
        )
        smoothness = self.edge_aware_smoothness(safe_prediction, image)
        total = (
            self.silog_weight * silog
            + self.log_l1_weight * log_l1
            + self.gradient_weight * gradient
            + self.smoothness_weight * smoothness
        )
        return total, {
            "silog": silog,
            "log_l1": log_l1,
            "gradient": gradient,
            "smoothness": smoothness,
        }


def move_batch(batch, device):
    tensor_keys = (
        "img", "nir", "depth", "valid_depth", "road", "batch_idx", "cls", "bboxes"
    )
    for key in tensor_keys:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def scalar_loss_items(items, use_l1_regression):
    """Normalize current Ultralytics dict losses and older tensor losses."""
    if isinstance(items, dict):
        normalized = {str(name): float(value.item()) for name, value in items.items()}
        if use_l1_regression and "dfl_loss" in normalized:
            normalized["l1_loss"] = normalized.pop("dfl_loss")
        return normalized
    third_name = "l1_loss" if use_l1_regression else "dfl_loss"
    names = ("box_loss", "cls_loss", third_name)
    return {
        name: float(items[index].item())
        for index, name in enumerate(names[: len(items)])
    }


def set_detector_trainable(model, trainable):
    for parameter in model.detector.parameters():
        parameter.requires_grad_(trainable)


def update_detection(totals, raw_output, batch, args):
    """Accumulate class-aware TP/FP/FN from native NMS-free YOLO26 output."""
    prediction = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    if prediction.ndim != 3 or prediction.shape[-1] != 6:
        raise ValueError(
            "Expected YOLO26 validation output shaped [B,N,6], got "
            f"{tuple(prediction.shape)}"
        )

    image_height, image_width = batch["img"].shape[-2:]
    for sample_index, sample in enumerate(prediction):
        sample = sample[sample[:, 4] >= args.val_conf][: args.val_max_det]
        predicted_boxes = [
            (
                int(row[5].item()),
                *row[:4].detach().cpu().tolist(),
                float(row[4].item()),
            )
            for row in sample
        ]
        selected = batch["batch_idx"] == sample_index
        classes = batch["cls"][selected, 0].detach().cpu().tolist()
        normalized = batch["bboxes"][selected].detach().cpu().tolist()
        gt_boxes = [
            (
                int(class_id),
                (cx - width / 2) * image_width,
                (cy - height / 2) * image_height,
                (cx + width / 2) * image_width,
                (cy + height / 2) * image_height,
                None,
            )
            for class_id, (cx, cy, width, height) in zip(classes, normalized)
        ]
        counts = match_detections(gt_boxes, predicted_boxes, args.val_eval_iou)
        for class_id, class_counts in counts.items():
            for name in ("tp", "fp", "fn"):
                value = class_counts[name]
                totals[f"det_{name}"] += value
                totals[f"det_class_{class_id}_{name}"] += value


def detection_metrics(totals):
    """Return micro and per-class detection Precision/Recall/F1."""
    metrics = {}
    prefixes = ["det", *[f"det_class_{index}" for index in CLASS_NAMES]]
    for prefix in prefixes:
        tp, fp, fn = (totals[f"{prefix}_{name}"] for name in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.update({
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
        })
    metrics.update({name: int(totals[name]) for name in ("det_tp", "det_fp", "det_fn")})
    return metrics


def augment_depth_nir(nir, dates, names, dropout_probability, mismatch_probability):
    """Teach the quality gate to reject missing or misaligned NIR inputs."""
    batch_size = nir.shape[0]
    augmented = nir.clone()
    mismatch_mask = torch.zeros(batch_size, dtype=torch.bool, device=nir.device)
    source_indices = torch.arange(batch_size, device=nir.device)

    date_groups = defaultdict(list)
    for index, date in enumerate(dates):
        date_groups[date].append(index)
    for indices in date_groups.values():
        if len(indices) < 2:
            continue
        offset = random.randint(1, len(indices) - 1)
        rolled = indices[offset:] + indices[:offset]
        for target, source in zip(indices, rolled):
            if names[target] != names[source] and random.random() < mismatch_probability:
                source_indices[target] = source
                mismatch_mask[target] = True
    augmented = augmented[source_indices]

    dropout_mask = torch.rand(batch_size, device=nir.device) < dropout_probability
    augmented[dropout_mask] = 0.0
    # A dropped sample is reported as dropout rather than mismatch.
    mismatch_mask &= ~dropout_mask
    return augmented, int(dropout_mask.sum().item()), int(mismatch_mask.sum().item())


def run_epoch(
    model,
    loader,
    det_criterion,
    depth_loss_fn,
    road_loss_fn,
    args,
    device,
    optimizer,
    scaler,
    detector_active=True,
):
    training = optimizer is not None
    model.train(training)
    if training and not detector_active:
        # Freeze parameters and running statistics during dense-head warm-up.
        model.detector.eval()
    totals = defaultdict(float)

    for batch in loader:
        batch = move_batch(batch, device)
        model_nir = batch["nir"]
        nir_dropout_count = 0
        nir_mismatch_count = 0
        if training and model.input_mode == "depth_nir_gated":
            model_nir, nir_dropout_count, nir_mismatch_count = augment_depth_nir(
                model_nir,
                batch["dates"],
                batch["names"],
                args.nir_dropout,
                args.nir_mismatch,
            )
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            detection, depth, road = model(batch["img"], model_nir)
            if training and not detector_active:
                detection_loss = depth.sum() * 0.0
                detection_items = {}
            else:
                detection_vector, detection_items = det_criterion(detection, batch)
                detection_loss = detection_vector.sum() / batch["img"].shape[0]
            depth_loss, depth_items = depth_loss_fn(
                depth, batch["depth"], batch["valid_depth"], batch["img"]
            )
            road_loss = road_loss_fn(road, batch["road"])
            total_loss = (
                detection_loss
                + args.depth_weight * depth_loss
                + args.road_weight * road_loss
            )

        if training:
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    "Non-finite training loss detected; aborting before optimizer step "
                    f"(det={float(detection_loss.detach()):.6g}, "
                    f"depth={float(depth_loss.detach()):.6g}, "
                    f"road={float(road_loss.detach()):.6g})"
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0, error_if_nonfinite=False
            )
            if not torch.isfinite(gradient_norm):
                if not scaler.is_enabled():
                    raise FloatingPointError(
                        "Non-finite gradient norm detected without AMP; "
                        "aborting before optimizer step"
                    )
                # Dynamic loss scaling is expected to overflow occasionally,
                # especially when detection is first enabled after dense-only
                # warm-up. GradScaler recorded the overflow in unscale_(), so
                # step() safely skips this update and update() lowers the scale.
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                print(
                    "AMP overflow: skipped optimizer step and reduced loss scale "
                    f"{previous_scale:g} -> {scaler.get_scale():g}"
                )
            else:
                scaler.step(optimizer)
                scaler.update()

        batch_size = batch["img"].shape[0]
        totals["samples"] += batch_size
        totals["nir_dropout_samples"] += nir_dropout_count
        totals["nir_mismatch_samples"] += nir_mismatch_count
        if (
            model.input_mode in ("scalar", "gated")
            and model.last_gate_mean is not None
        ):
            totals["gate_mean"] += float(model.last_gate_mean.item()) * batch_size
        if (
            model.input_mode in ("depth_nir", "depth_nir_gated")
            and model.last_depth_nir_alpha is not None
        ):
            totals["nir_alpha_p1"] += float(model.last_depth_nir_alpha[0]) * batch_size
            totals["nir_alpha_p2"] += float(model.last_depth_nir_alpha[1]) * batch_size
        if (
            model.input_mode == "depth_nir_gated"
            and model.last_depth_nir_gate_mean is not None
        ):
            totals["nir_gate_p1"] += float(
                model.last_depth_nir_gate_mean[:, 0].mean()
            ) * batch_size
            totals["nir_gate_p2"] += float(
                model.last_depth_nir_gate_mean[:, 1].mean()
            ) * batch_size
        for name, value in (
            ("loss", total_loss),
            ("det_loss", detection_loss),
            ("depth_loss", depth_loss),
            ("road_loss", road_loss),
        ):
            totals[name] += float(value.item()) * batch_size
        for name, value in depth_items.items():
            totals[f"depth_{name}"] += float(value.item()) * batch_size
        use_l1 = int(model.detector.model[-1].reg_max) == 1
        for name, value in scalar_loss_items(detection_items, use_l1).items():
            totals[name] += value * batch_size

        update_depth(totals, depth.detach(), batch["depth"], batch["valid_depth"])
        update_road(totals, road.detach(), batch["road"])
        if not training:
            update_detection(totals, detection, batch, args)

    samples = max(1, totals["samples"])
    pixels = max(1, totals["pixels"])
    tp, fp, fn = totals["road_tp"], totals["road_fp"], totals["road_fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    metrics = {
        "loss": totals["loss"] / samples,
        "det_loss": totals["det_loss"] / samples,
        "depth_loss": totals["depth_loss"] / samples,
        "depth_silog": totals["depth_silog"] / samples,
        "depth_log_l1": totals["depth_log_l1"] / samples,
        "depth_gradient": totals["depth_gradient"] / samples,
        "depth_smoothness": totals["depth_smoothness"] / samples,
        "road_loss": totals["road_loss"] / samples,
        "box_loss": totals["box_loss"] / samples,
        "cls_loss": totals["cls_loss"] / samples,
        # YOLO26 reg_max=1: this is normalized L1 regression, not DFL.
        "l1_loss": totals["l1_loss"] / samples,
        "abs_rel": totals["abs_rel_sum"] / pixels,
        "rmse": math.sqrt(totals["square_sum"] / pixels),
        "delta1": totals["delta1_sum"] / pixels,
        "road_iou": tp / max(1, tp + fp + fn),
        "road_precision": precision,
        "road_recall": recall,
        "road_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    if model.input_mode in ("scalar", "gated"):
        metrics["gate_mean"] = totals["gate_mean"] / samples
    if model.input_mode in ("depth_nir", "depth_nir_gated"):
        metrics["nir_alpha_p1"] = totals["nir_alpha_p1"] / samples
        metrics["nir_alpha_p2"] = totals["nir_alpha_p2"] / samples
    if model.input_mode == "depth_nir_gated":
        metrics["nir_gate_p1"] = totals["nir_gate_p1"] / samples
        metrics["nir_gate_p2"] = totals["nir_gate_p2"] / samples
        metrics["nir_dropout_fraction"] = totals["nir_dropout_samples"] / samples
        metrics["nir_mismatch_fraction"] = totals["nir_mismatch_samples"] / samples
    metrics.update(detection_metrics(totals) if not training else {
        name: float("nan") for name in (
            "det_precision", "det_recall", "det_f1", "det_tp", "det_fp", "det_fn",
            *[
                f"det_class_{index}_{metric}"
                for index in CLASS_NAMES
                for metric in ("precision", "recall", "f1")
            ],
        )
    })
    return metrics


def make_loaders(args, device):
    image_size = resolve_image_size(args)
    train_set = ThreeTaskDataset(
        args.dataset_root, "train", image_size, args.min_depth, args.max_depth, True, **dataset_options(args)
    )
    val_set = ThreeTaskDataset(
        args.dataset_root, "test", image_size, args.min_depth, args.max_depth, False, **dataset_options(args)
    )

    date_counts = defaultdict(int)
    for row in train_set.rows:
        date_counts[row["date"]] += 1
    weights = [1.0 / date_counts[row["date"]] for row in train_set.rows]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    common = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=args.batch, sampler=sampler, **common
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False, **common
    )
    print(
        f"train={len(train_set)} test={len(val_set)} "
        f"train_dates={dict(date_counts)} device={device}"
    )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Checkpoints and history
# -----------------------------------------------------------------------------


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, args):
    torch.save(
        {
            "epoch": epoch,
            "architecture": (
                "YOLO26 RGB detector/Road with quality-gated depth-only P1-P2 NIR fusion MTL"
                if model.input_mode == "depth_nir_gated"
                else "YOLO26 RGB detector/Road with depth-only P1-P2 NIR fusion MTL"
                if model.input_mode == "depth_nir"
                else (
                    "YOLO26 RGB-NIR P1 dual-stem scalar-fusion MTL"
                    if model.input_mode == "scalar"
                    else (
                        "YOLO26 RGB-NIR P1 spatial-channel gated-fusion MTL"
                        if model.input_mode == "gated"
                        else (
                            "YOLO26 RGB-NIR four-channel early-fusion MTL"
                            if model.input_mode == "rgbn"
                            else "YOLO26 RGB-NIR HSV channel-attention image fusion MTL"
                        )
                    )
                )
            ),
            "architecture_version": model.architecture_version,
            "input_mode": model.input_mode,
            "feature_indices": model.feature_indices,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best": best,
            "args": vars(args),
            "ultralytics_version": ultralytics_version,
        },
        path,
    )


def save_detector(path, model):
    """Save an Ultralytics-compatible detection-only YOLO26 checkpoint."""
    detector = deepcopy(model.detector).to("cpu").half().eval()
    detector.criterion = None
    torch.save(
        {
            "model": detector,
            "ema": None,
            "optimizer": None,
            "train_args": vars(detector.args),
            "date": datetime.now().astimezone().isoformat(),
            "version": ultralytics_version,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        },
        path,
    )


def append_history(path, epoch, seconds, train_metrics, val_metrics, start_epoch):
    metric_names = list(train_metrics)
    fields = ["epoch", "seconds"] + [
        f"{split}_{name}" for split in ("train", "val") for name in metric_names
    ]
    row = {"epoch": epoch, "seconds": seconds}
    row.update({f"train_{name}": value for name, value in train_metrics.items()})
    row.update({f"val_{name}": value for name, value in val_metrics.items()})
    write_header = not path.exists() or (start_epoch == 1 and epoch == 1)
    with path.open("w" if write_header else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    validate_environment(args)
    image_size = resolve_image_size(args)
    set_random_seed(args.seed)

    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics_yolo26")
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    device = select_device(args.device)

    detector = build_yolo26_detector(args, device)
    image_fusion = (
        build_pixel_aligned_image_fusion(args, device)
        if args.input_mode == "hsvnet"
        else None
    )
    feature_fusion = (
        build_scalar_feature_fusion(detector, device)
        if args.input_mode == "scalar"
        else (
            build_gated_feature_fusion(detector, device)
            if args.input_mode == "gated"
            else None
        )
    )
    nir_encoder = (
        build_depth_nir_encoder(
            detector, device, quality_gated=args.input_mode == "depth_nir_gated"
        )
        if args.input_mode in ("depth_nir", "depth_nir_gated")
        else None
    )
    feature_indices, feature_channels = discover_dense_features(
        detector, image_size, device
    )
    model = Yolo26DepthDetRoad(
        detector,
        feature_indices,
        feature_channels,
        args.min_depth,
        args.max_depth,
        args.depth_init,
        image_fusion=image_fusion,
        feature_fusion=feature_fusion,
        nir_encoder=nir_encoder,
        input_mode=args.input_mode,
    ).to(device)

    parameter_groups = [
        {"params": model.detector.parameters(), "lr": args.lr},
        {
            "params": model.dense_head.parameters(),
            "lr": args.lr * args.dense_lr_mult,
        },
    ]
    if model.image_fusion is not None:
        parameter_groups.append(
            {
                "params": model.image_fusion.parameters(),
                "lr": args.lr * args.fusion_lr_mult,
            }
        )
    if model.feature_fusion is not None:
        parameter_groups.append(
            {
                "params": model.feature_fusion.parameters(),
                "lr": args.lr * args.feature_fusion_lr_mult,
            }
        )
    if model.nir_encoder is not None:
        parameter_groups.append(
            {
                "params": model.nir_encoder.parameters(),
                "lr": args.lr * args.feature_fusion_lr_mult,
            }
        )
    optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    start_epoch = 1
    best = {"joint": float("inf"), "depth": float("inf"), "road": -1.0}

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_mode = checkpoint.get("input_mode") or checkpoint.get("args", {}).get(
            "input_mode", "hsvnet"
        )
        if checkpoint_mode != args.input_mode:
            raise ValueError(
                f"Resume checkpoint input mode {checkpoint_mode!r} does not match "
                f"requested mode {args.input_mode!r}"
            )
        if tuple(checkpoint.get("feature_indices", ())) != tuple(feature_indices):
            raise ValueError(
                "Resume checkpoint feature indices do not match this YOLO26 graph: "
                f"{checkpoint.get('feature_indices')} vs {feature_indices}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    detection_criterion = model.detector.init_criterion()
    # YOLO26 progressively shifts weight from one-to-many to one-to-one once per epoch.
    completed_joint_epochs = max(0, start_epoch - 1 - args.head_warmup_epochs)
    for _ in range(completed_joint_epochs):
        if hasattr(detection_criterion, "update"):
            detection_criterion.update()

    train_loader, val_loader = make_loaders(args, device)
    depth_loss_fn = SparseDepthLoss(args)
    road_loss_fn = RoadLoss()
    history_path = output_dir / "history.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        begin = time.time()
        detector_active = epoch > args.head_warmup_epochs
        set_detector_trainable(model, detector_active)
        phase = "joint" if detector_active else "fusion+dense warm-up"
        print(f"epoch {epoch:03d} phase={phase}")
        train_metrics = run_epoch(
            model,
            train_loader,
            detection_criterion,
            depth_loss_fn,
            road_loss_fn,
            args,
            device,
            optimizer,
            scaler,
            detector_active=detector_active,
        )
        if detector_active and hasattr(detection_criterion, "update"):
            detection_criterion.update()
        val_metrics = run_epoch(
            model,
            val_loader,
            detection_criterion,
            depth_loss_fn,
            road_loss_fn,
            args,
            device,
            optimizer=None,
            scaler=scaler,
            detector_active=True,
        )
        scheduler.step()

        append_history(
            history_path,
            epoch,
            time.time() - begin,
            train_metrics,
            val_metrics,
            start_epoch,
        )
        joint = (
            val_metrics["det_loss"]
            + args.depth_weight * val_metrics["depth_loss"]
            + args.road_weight * val_metrics["road_loss"]
        )
        improved = {
            "joint": joint < best["joint"],
            "depth": val_metrics["abs_rel"] < best["depth"],
            "road": val_metrics["road_iou"] > best["road"],
        }
        if improved["joint"]:
            best["joint"] = joint
        if improved["depth"]:
            best["depth"] = val_metrics["abs_rel"]
        if improved["road"]:
            best["road"] = val_metrics["road_iou"]

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best,
            args,
        )
        if model.input_mode not in ("scalar", "gated"):
            detector_name = {
                "depth_nir": "last_detector_rgb.pt",
                "depth_nir_gated": "last_detector_rgb.pt",
                "rgbn": "last_detector_rgbn.pt",
                "hsvnet": "last_detector_on_fused_rgb.pt",
            }[model.input_mode]
            save_detector(output_dir / detector_name, model)
        for name in ("joint", "depth", "road"):
            if improved[name]:
                save_checkpoint(
                    output_dir / f"best_{name}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best,
                    args,
                )

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        print(
            "  Detection | "
            f"P={val_metrics['det_precision']:.4f} "
            f"R={val_metrics['det_recall']:.4f} "
            f"F1={val_metrics['det_f1']:.4f} "
            f"TP/FP/FN={val_metrics['det_tp']}/{val_metrics['det_fp']}/{val_metrics['det_fn']} "
            f"loss={val_metrics['det_loss']:.4f}"
        )
        class_metrics = " | ".join(
            f"{CLASS_NAMES[index]}: "
            f"P={val_metrics[f'det_class_{index}_precision']:.3f} "
            f"R={val_metrics[f'det_class_{index}_recall']:.3f} "
            f"F1={val_metrics[f'det_class_{index}_f1']:.3f}"
            for index in CLASS_NAMES
        )
        print(f"    per class | {class_metrics}")
        print(
            "  Depth     | "
            f"AbsRel={val_metrics['abs_rel']:.4f} "
            f"RMSE={val_metrics['rmse']:.3f} "
            f"delta1={val_metrics['delta1']:.4f} "
            f"loss={val_metrics['depth_loss']:.4f}"
        )
        print(
            "  Road      | "
            f"IoU={val_metrics['road_iou']:.4f} "
            f"P={val_metrics['road_precision']:.4f} "
            f"R={val_metrics['road_recall']:.4f} "
            f"F1={val_metrics['road_f1']:.4f} "
            f"loss={val_metrics['road_loss']:.4f}"
        )
        if model.input_mode in ("scalar", "gated"):
            print(
                "  RGB-NIR fusion weight | "
                f"train_mean={train_metrics['gate_mean']:.4f} "
                f"val_mean={val_metrics['gate_mean']:.4f}"
            )
        if model.input_mode in ("depth_nir", "depth_nir_gated"):
            print(
                "  Depth NIR weights | "
                f"P1={val_metrics['nir_alpha_p1']:.5f} "
                f"P2={val_metrics['nir_alpha_p2']:.5f}"
            )
        if model.input_mode == "depth_nir_gated":
            print(
                "  Depth NIR gates   | "
                f"P1={val_metrics['nir_gate_p1']:.4f} "
                f"P2={val_metrics['nir_gate_p2']:.4f} "
                f"train dropout/mismatch="
                f"{train_metrics['nir_dropout_fraction']:.3f}/"
                f"{train_metrics['nir_mismatch_fraction']:.3f}"
            )


if __name__ == "__main__":
    main()
