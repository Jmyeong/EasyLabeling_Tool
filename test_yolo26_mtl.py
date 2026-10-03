#!/usr/bin/env python3
"""Evaluate YOLO26 detection + depth + Road segmentation on the DY test sets.

YOLO26 uses its native one-to-one, NMS-free detection output. Metrics and
visualizations intentionally match the existing YOLO11 three-task test.
"""

import argparse
from dated_dataset import add_dataset_arguments, dataset_options
import json
import os
from collections import defaultdict
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from ultralytics.utils.torch_utils import get_flops

from train import ThreeTaskDataset, collate, resolve_image_size
from train_yolo26_mtl import (
    Yolo26DepthDetRoad,
    build_depth_nir_encoder,
    build_gated_feature_fusion,
    build_pixel_aligned_image_fusion,
    build_scalar_feature_fusion,
    build_yolo26_detector,
    discover_dense_features,
    validate_environment,
)
from visualize_yolo_test import match_detections

# Reuse the established metric, visualization, and reporting functions so the
# YOLO11 and YOLO26 result files have the same schema.
from test import (
    finish_depth_metrics,
    finish_detection_metrics,
    finish_road_metrics,
    depth_evaluation_mask,
    filter_boxes_by_gt_depth,
    get_gt_boxes,
    make_overview,
    move_batch_to_device,
    print_reports,
    save_csv,
    save_html,
    save_six_panel,
    save_task_visualizations,
    select_device,
    synchronize,
    update_depth_counts,
    update_detection_counts,
    update_road_counts,
    warm_up_model,
)


PROJECT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Arguments and checkpoint loading
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test YOLO26 detection + metric depth + Road segmentation"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_DIR / "runs" / "yolo26s_rgb_nir_gated_fusion_mtl" / "best_depth.pt"
        ),
    )
    parser.add_argument(
        "--model-template",
        default="",
        help=(
            "Optional matching YOLO26 .pt or .yaml used to rebuild the graph. "
            "Empty uses the model recorded in the joint checkpoint."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "results" / "yolo26s_rgb_nir_gated_fusion_mtl_best_depth"),
    )
    parser.add_argument(
        "--input-mode",
        choices=(
            "auto", "depth_nir_gated", "depth_nir", "scalar", "gated", "rgbn", "hsvnet"
        ),
        default="auto",
        help="auto reads the mode from the checkpoint; legacy fusion checkpoints resolve to hsvnet",
    )
    parser.add_argument(
        "--fusion-repo",
        default="",
        help="Override Pixel-aligned RGB-NIR Stereo repository recorded in checkpoint",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        default="",
        help="Override HSVNet image-fusion checkpoint recorded in checkpoint",
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
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all test samples")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument(
        "--nir-ablation",
        choices=("normal", "zero", "shuffle"),
        default="normal",
        help="Use aligned NIR, zero NIR, or batch-shuffled NIR for causal evaluation",
    )

    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--eval-iou", type=float, default=0.30)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=35.0)
    parser.add_argument(
        "--eval-depth-in",
        "--eval_depth_in",
        type=float,
        default=None,
        help="Evaluate detection and depth only at GT depth <= this distance in meters",
    )
    parser.add_argument("--road-threshold", type=float, default=0.50)

    parser.add_argument("--sheet-count", type=int, default=36)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument(
        "--save-per-sample",
        "--save_per_sample",
        action="store_true",
        help=(
            "Save per_sample_metrics.csv and aligned RGB/NIR pair images, including "
            "exposure statistics and per-image task metrics"
        ),
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def validate_test_environment(args):
    # Reuse the training environment check without exposing irrelevant options.
    environment_args = SimpleNamespace(
        height=args.height,
        width=args.width,
        imgsz=args.imgsz,
        device=args.device,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_init=8.0,
        head_warmup_epochs=0,
        input_mode=args.input_mode,
        fusion_repo=args.fusion_repo,
        fusion_checkpoint=args.fusion_checkpoint,
        nir_dropout=0.0,
        nir_mismatch=0.0,
    )
    validate_environment(environment_args)


def load_model(args, device):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"YOLO26 joint checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if not str(checkpoint.get("architecture", "")).startswith("YOLO26"):
        raise ValueError(
            "This is not a train_yolo26_mtl.py checkpoint: "
            f"architecture={checkpoint.get('architecture')!r}"
        )

    checkpoint_mode = checkpoint.get("input_mode") or checkpoint_args.get("input_mode")
    if checkpoint_mode is None:
        checkpoint_mode = "hsvnet"  # checkpoints produced before input_mode was recorded
    if args.input_mode != "auto" and args.input_mode != checkpoint_mode:
        raise ValueError(
            f"Requested input mode {args.input_mode!r} does not match checkpoint "
            f"mode {checkpoint_mode!r}"
        )
    args.input_mode = checkpoint_mode

    image_size = resolve_image_size(args)
    template = args.model_template or checkpoint_args.get("model", "yolo26s.pt")
    build_args = SimpleNamespace(
        model=template,
        dataset_root=args.dataset_root,
        epochs=int(checkpoint_args.get("epochs", 100)),
        height=image_size[0],
        width=image_size[1],
        imgsz=None,
        input_mode=checkpoint_mode,
        fusion_repo=(
            args.fusion_repo
            or checkpoint_args.get(
                "fusion_repo",
                "external/Pixel_aligned_RGB_NIR_Stereo",
            )
        ),
        fusion_checkpoint=(
            args.fusion_checkpoint
            or checkpoint_args.get(
                "fusion_checkpoint",
                "external/Pixel_aligned_RGB_NIR_Stereo/"
                "weights/model_image_fusion.pth",
            )
        ),
    )
    # Fill resolved paths before validation so empty CLI overrides remain useful.
    args.fusion_repo = build_args.fusion_repo
    args.fusion_checkpoint = build_args.fusion_checkpoint
    detector = build_yolo26_detector(build_args, device)
    image_fusion = (
        build_pixel_aligned_image_fusion(build_args, device)
        if checkpoint_mode == "hsvnet"
        else None
    )
    feature_fusion = (
        build_scalar_feature_fusion(detector, device)
        if checkpoint_mode == "scalar"
        else (
            build_gated_feature_fusion(detector, device)
            if checkpoint_mode == "gated"
            else None
        )
    )
    nir_encoder = (
        build_depth_nir_encoder(
            detector, device, quality_gated=checkpoint_mode == "depth_nir_gated"
        )
        if checkpoint_mode in ("depth_nir", "depth_nir_gated")
        else None
    )
    detector.model[-1].max_det = args.max_det
    saved_indices = tuple(checkpoint.get("feature_indices", ()))
    feature_indices, feature_channels = discover_dense_features(
        detector,
        image_size,
        device,
        include_p1=len(saved_indices) != 4,
    )
    if saved_indices and saved_indices != tuple(feature_indices):
        raise ValueError(
            "Checkpoint/template feature mismatch: "
            f"saved={saved_indices}, discovered={feature_indices}"
        )

    min_depth = float(checkpoint_args.get("min_depth", args.min_depth))
    max_depth = float(checkpoint_args.get("max_depth", args.max_depth))
    initial_depth = float(checkpoint_args.get("depth_init", 8.0))
    model = Yolo26DepthDetRoad(
        detector,
        feature_indices,
        feature_channels,
        min_depth=min_depth,
        max_depth=max_depth,
        initial_depth=initial_depth,
        image_fusion=image_fusion,
        feature_fusion=feature_fusion,
        nir_encoder=nir_encoder,
        input_mode=checkpoint_mode,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


# -----------------------------------------------------------------------------
# Native YOLO26 NMS-free output
# -----------------------------------------------------------------------------


def decode_yolo26_detections(raw_output, args):
    """Convert native [x1,y1,x2,y2,confidence,class] output to test tuples."""
    prediction = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    if prediction.ndim != 3 or prediction.shape[-1] != 6:
        raise ValueError(
            "Expected YOLO26 end-to-end output shaped [B,N,6], got "
            f"{tuple(prediction.shape)}"
        )

    decoded = []
    for sample in prediction:
        sample = sample[sample[:, 4] >= args.conf]
        sample = sample[: args.max_det]
        decoded.append(
            [
                (
                    int(row[5].item()),
                    *row[:4].detach().cpu().tolist(),
                    float(row[4].item()),
                )
                for row in sample
            ]
        )
    return decoded


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------


def safe_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def intensity_statistics(values, prefix):
    """Summarize normalized sensor intensities inside the non-letterboxed area."""
    values = values.float().flatten()
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_dark_fraction": 0.0,
            f"{prefix}_bright_fraction": 0.0,
        }
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.95], device=values.device))
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_p05": float(quantiles[0].item()),
        f"{prefix}_p95": float(quantiles[1].item()),
        f"{prefix}_dark_fraction": float((values < 0.15).float().mean().item()),
        f"{prefix}_bright_fraction": float((values > 0.85).float().mean().item()),
    }


def save_rgb_nir_pair(rgb, nir, output_path):
    """Save the aligned network inputs side by side for quick visual inspection."""
    rgb_array = (
        rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    nir_array = (
        nir.detach().cpu().squeeze(0).numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    nir_array = np.repeat(nir_array[..., None], 3, axis=2)
    gap = np.full((rgb_array.shape[0], 8, 3), 255, dtype=np.uint8)
    pair = np.concatenate((rgb_array, gap, nir_array), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pair).save(output_path, quality=92, subsampling=0)


def build_per_sample_row(
    batch,
    sample_index,
    image_counts,
    predicted_depth,
    road_logits,
    metric_depth_valid,
    rendered,
    pair_path,
    args,
    nir_gate=None,
):
    det_tp = sum(int(counts["tp"]) for counts in image_counts.values())
    det_fp = sum(int(counts["fp"]) for counts in image_counts.values())
    det_fn = sum(int(counts["fn"]) for counts in image_counts.values())
    det_precision, det_recall, det_f1 = safe_prf(det_tp, det_fp, det_fn)

    target_depth = batch["depth"][sample_index][metric_depth_valid].clamp_min(1e-4)
    sample_prediction = predicted_depth[sample_index][metric_depth_valid].clamp_min(1e-4)
    if target_depth.numel():
        difference = sample_prediction - target_depth
        ratio = torch.maximum(sample_prediction / target_depth, target_depth / sample_prediction)
        depth_abs_rel = float((difference.abs() / target_depth).mean().item())
        depth_rmse = float(difference.square().mean().sqrt().item())
        depth_delta1 = float((ratio < 1.25).float().mean().item())
    else:
        depth_abs_rel = depth_rmse = depth_delta1 = 0.0

    road_target = batch["road"][sample_index]
    road_valid = road_target != 255
    road_prediction = road_logits[sample_index][road_valid].sigmoid() >= args.road_threshold
    road_ground_truth = road_target[road_valid] == 1
    road_tp = int((road_prediction & road_ground_truth).sum().item())
    road_fp = int((road_prediction & ~road_ground_truth).sum().item())
    road_fn = int((~road_prediction & road_ground_truth).sum().item())
    road_precision, road_recall, road_f1 = safe_prf(road_tp, road_fp, road_fn)
    road_union = road_tp + road_fp + road_fn

    # The road ignore mask also marks the letterbox padding. Statistics are
    # calculated from the original image region, not from padded pixels.
    rgb = batch["img"][sample_index]
    nir = batch["nir"][sample_index, 0]
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    row = {
        "name": batch["names"][sample_index],
        "date": batch["dates"][sample_index],
        "nir_source_name": batch["nir_names"][sample_index],
        "nir_ablation": args.nir_ablation,
        **intensity_statistics(luminance[road_valid], "rgb_luma"),
        **intensity_statistics(nir[road_valid], "nir"),
        "det_tp": det_tp,
        "det_fp": det_fp,
        "det_fn": det_fn,
        "det_precision": det_precision,
        "det_recall": det_recall,
        "det_f1": det_f1,
        "depth_valid_pixels": int(target_depth.numel()),
        "depth_abs_rel": depth_abs_rel,
        "depth_rmse": depth_rmse,
        "depth_delta1": depth_delta1,
        "road_valid_pixels": int(road_valid.sum().item()),
        "road_tp": road_tp,
        "road_fp": road_fp,
        "road_fn": road_fn,
        "road_iou": road_tp / road_union if road_union else 0.0,
        "road_precision": road_precision,
        "road_recall": road_recall,
        "road_f1": road_f1,
        "rendered_path": rendered,
        "rgb_nir_pair_path": str(pair_path),
    }
    if nir_gate is not None:
        row["nir_gate_p1"] = float(nir_gate[0].item())
        row["nir_gate_p2"] = float(nir_gate[1].item())
    for class_id, counts in sorted(image_counts.items()):
        for key in ("tp", "fp", "fn"):
            row[f"det_class_{class_id}_{key}"] = int(counts[key])
    return row


def evaluate(model, loader, device, output_root, args):
    detection_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    depth_counts = defaultdict(lambda: defaultdict(float))
    road_counts = defaultdict(lambda: defaultdict(float))
    entries = []
    per_sample_rows = []
    inference_seconds = 0.0
    measured_batches = 0
    processed = 0

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            model_nir = batch["nir"]
            if args.nir_ablation == "zero":
                model_nir = torch.zeros_like(model_nir)

            # Only the shared three-task forward is timed. Loading, decoding,
            # metric calculation, and visualization are excluded.
            synchronize(device)
            forward_start = time.perf_counter()
            raw_detection, predicted_depth, road_logits = model(
                batch["img"], model_nir
            )
            batch_nir_gates = model.last_depth_nir_gate_mean
            synchronize(device)
            inference_seconds += time.perf_counter() - forward_start
            measured_batches += 1

            predicted_boxes = decode_yolo26_detections(raw_detection, args)
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
                image_counts = match_detections(gt_boxes, prediction, args.eval_iou)

                for scope in ("all", date):
                    update_detection_counts(detection_counts, scope, image_counts)
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
                if args.save_per_sample:
                    pair_path = output_root / "per_sample" / "rgb_nir" / date / f"{name}.jpg"
                    save_rgb_nir_pair(
                        batch["img"][sample_index], batch["nir"][sample_index], pair_path
                    )
                    per_sample_rows.append(
                        build_per_sample_row(
                            batch,
                            sample_index,
                            image_counts,
                            predicted_depth,
                            road_logits,
                            metric_depth_valid,
                            rendered,
                            pair_path,
                            args,
                            (
                                batch_nir_gates[sample_index]
                                if batch_nir_gates is not None
                                else None
                            ),
                        )
                    )
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
        per_sample_rows,
    )


# -----------------------------------------------------------------------------
# Entry point and reports
# -----------------------------------------------------------------------------


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
    validate_test_environment(args)
    image_size = resolve_image_size(args)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics_yolo26")
    )
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
        **dataset_options(args),
    )
    if args.limit > 0:
        dataset.rows = dataset.rows[: args.limit]
    if args.nir_ablation == "shuffle":
        if len(dataset) < 2:
            raise ValueError("--nir-ablation shuffle requires at least two samples")
        shuffled, groups = dataset.set_grouped_nir_shuffle()
        print(
            "NIR ablation: deterministic within-date/sequence shuffle "
            f"(shuffled={shuffled}, groups={groups})"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )

    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(
        f"epoch={checkpoint.get('epoch')} test={len(dataset)} "
        f"input={image_size[0]}x{image_size[1]} device={device}"
    )
    gflops = float(get_flops(model, imgsz=list(image_size)))
    warm_up_model(model, device, image_size, args.warmup_runs)
    entries, detection, depth, road, performance, per_sample_rows = evaluate(
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
            "nms_free": True,
        }
    )

    save_csv(output_root / "detection_metrics.csv", detection)
    save_csv(output_root / "depth_metrics.csv", depth)
    save_csv(output_root / "road_metrics.csv", road)
    save_csv(output_root / "inference_metrics.csv", [performance])
    if args.save_per_sample:
        save_csv(output_root / "per_sample_metrics.csv", per_sample_rows)
    summary = {
        "architecture": checkpoint.get("architecture"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "settings": vars(args),
        "detection": detection,
        "depth": depth,
        "road": road,
        "performance": performance,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not args.no_images:
        for date in sorted({entry["date"] for entry in entries}):
            make_overview(
                [entry for entry in entries if entry["date"] == date],
                output_root / f"overview_{date}_test_yolo26_3task.jpg",
                args,
            )
    save_html(output_root, entries, detection, depth, road, performance, args)
    print_reports(detection, depth, road, performance)
    print(f"\noutput: {output_root}")
    print(f"index: {output_root / 'index.html'}")
    print(f"summary: {output_root / 'summary.json'}")
    if args.save_per_sample:
        print(f"per sample: {output_root / 'per_sample_metrics.csv'}")


if __name__ == "__main__":
    main()
