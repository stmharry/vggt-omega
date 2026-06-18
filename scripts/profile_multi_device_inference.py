#!/usr/bin/env python
"""Profile experimental multi-device VGGT-Omega inference paths."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


OUTPUT_KEYS = (
    "pose_enc",
    "depth",
    "depth_conf",
    "camera_and_register_tokens",
)


class ProfileOutOfMemoryError(RuntimeError):
    def __init__(self, message: str, summary: dict[str, Any]) -> None:
        super().__init__(message)
        self.summary = summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VGGT-Omega single-GPU and experimental GPU-parallel inference "
            "for memory/parity investigation."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "single",
            "head-parallel",
            "memory-parallel",
            "balanced-memory-parallel",
            "pipeline-memory-parallel",
            "fsdp",
            "compare",
            "capacity",
            "plan",
        ),
        default="single",
        help="Inference path to profile. compare runs single and the selected --compare-mode sequentially.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        help="Optional VGGT-Omega checkpoint path. Omit to profile randomly initialized weights.",
    )
    parser.add_argument(
        "--image-dir",
        type=pathlib.Path,
        help="Directory of input images. If omitted, synthetic random images are used.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        help="Limit image-dir inputs to this many sorted images.",
    )
    parser.add_argument(
        "--synthetic-frames",
        type=int,
        default=3,
        help="Synthetic frame count used when --image-dir is omitted.",
    )
    parser.add_argument(
        "--synthetic-height",
        type=int,
        default=512,
        help="Synthetic image height.",
    )
    parser.add_argument(
        "--synthetic-width",
        type=int,
        default=512,
        help="Synthetic image width.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=512,
        help="Resolution passed to load_and_preprocess_images.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("balanced", "max_size"),
        default="balanced",
        help="Preprocessing resize mode.",
    )
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated CUDA device indices, e.g. 0 or 0,1.",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        help="Optional path for a JSON summary.",
    )
    parser.add_argument(
        "--parity-atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for compare mode.",
    )
    parser.add_argument(
        "--parity-rtol",
        type=float,
        default=5e-3,
        help="Relative tolerance for compare mode.",
    )
    parser.add_argument(
        "--capacity-mode",
        choices=("single", "head-parallel", "memory-parallel", "balanced-memory-parallel", "pipeline-memory-parallel"),
        default="memory-parallel",
        help="Inference path used by capacity mode.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("head-parallel", "memory-parallel", "balanced-memory-parallel", "pipeline-memory-parallel"),
        default="memory-parallel",
        help="GPU-parallel inference path compared against single in compare mode.",
    )
    parser.add_argument(
        "--split-block",
        type=int,
        default=23,
        help="Aggregator block index where balanced-memory-parallel moves live tokens to the secondary device.",
    )
    parser.add_argument(
        "--stage-splits",
        default="8,16",
        help="Comma-separated aggregator split blocks for pipeline-memory-parallel, e.g. 8,16.",
    )
    parser.add_argument(
        "--cache-device-index",
        type=int,
        default=3,
        help="Index into --devices used for cached outputs and heads in pipeline-memory-parallel.",
    )
    parser.add_argument(
        "--cache-device",
        help=(
            "Explicit cache device for pipeline-memory-parallel. Use 'cpu' to offload cached "
            "aggregator layers and keep every CUDA device available as a pipeline stage. "
            "When omitted, --cache-device-index preserves the original behavior."
        ),
    )
    parser.add_argument(
        "--head-device-index",
        type=int,
        help=(
            "Index into --devices used for camera/depth/text heads in pipeline-memory-parallel. "
            "Defaults to the cache GPU, or the last CUDA device when --cache-device=cpu."
        ),
    )
    parser.add_argument(
        "--patch-embed-chunk-size",
        type=int,
        help="Optional frame chunk size for patch embedding to reduce primary-device transient memory.",
    )
    parser.add_argument(
        "--input-device",
        choices=("primary", "cpu"),
        default="primary",
        help="Where to keep loaded inputs before model forward. 'cpu' lets the model stage image chunks onto CUDA.",
    )
    parser.add_argument(
        "--offload-outputs-to-cpu",
        action="store_true",
        help="Move returned tensors to CPU during pipeline-memory-parallel inference to reduce retained GPU output memory.",
    )
    parser.add_argument(
        "--query-blockwise-devices",
        help=(
            "Comma-separated visible CUDA indices used for exact blockwise query-sharded "
            "aggregator global inter-frame attention in pipeline-memory-parallel."
        ),
    )
    parser.add_argument(
        "--query-block-size",
        type=int,
        default=2048,
        help="Query rows per block for --query-blockwise-devices.",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "efficient", "flash_math", "math"),
        default="auto",
        help=(
            "Scaled dot-product attention backend policy. 'efficient' disables flash, "
            "math, and cuDNN SDPA; 'flash_math' enables flash with math fallback and "
            "disables memory-efficient and cuDNN SDPA; 'math' disables flash, "
            "memory-efficient, and cuDNN SDPA."
        ),
    )
    parser.add_argument(
        "--frame-counts",
        default="50,100,200,300,400",
        help="Comma-separated frame counts for capacity mode.",
    )
    parser.add_argument(
        "--auto-plan",
        action="store_true",
        help=(
            "Choose pipeline-memory-parallel placement from device inventory and memory estimates. "
            "Capacity mode also runs parity probes before large-frame tests unless --skip-auto-plan-parity is set."
        ),
    )
    parser.add_argument(
        "--auto-plan-query-shards",
        choices=("auto", "off"),
        default="auto",
        help="Whether auto-plan may enable exact query-sharded aggregator attention.",
    )
    parser.add_argument(
        "--auto-plan-sdpa-backends",
        default="flash_math,efficient,auto,math",
        help="Comma-separated SDPA backend policies tried by auto-plan parity probes.",
    )
    parser.add_argument(
        "--auto-plan-parity-frames",
        default="3,10,25",
        help="Comma-separated frame counts used by auto-plan capacity parity probes.",
    )
    parser.add_argument(
        "--skip-auto-plan-parity",
        action="store_true",
        help="Skip auto-plan parity probes in capacity mode and use the first planned backend candidate directly.",
    )
    parser.add_argument(
        "--auto-plan-probe-isolation",
        choices=("subprocess", "in-process"),
        default="subprocess",
        help=(
            "How auto-plan parity probes are executed. Subprocess isolation keeps CUDA illegal-access "
            "or backend crashes from poisoning the parent capacity run."
        ),
    )
    return parser.parse_args()


def parse_cuda_devices(value: str) -> list[torch.device]:
    indices = [part.strip() for part in value.split(",") if part.strip()]
    if not indices:
        raise ValueError("--devices must contain at least one CUDA index")
    devices = [torch.device(f"cuda:{int(index)}") for index in indices]
    visible = torch.cuda.device_count()
    missing = [str(device) for device in devices if device.index is None or device.index >= visible]
    if missing:
        raise RuntimeError(f"Requested CUDA devices are not visible: {missing}; visible_count={visible}")
    return devices


def parse_frame_counts(value: str) -> list[int]:
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts:
        raise ValueError("--frame-counts must contain at least one count")
    if any(count <= 0 for count in counts):
        raise ValueError("--frame-counts values must be positive")
    return counts


def parse_stage_splits(value: str) -> list[int]:
    splits = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("--stage-splits must contain at least one split")
    return splits


def parse_optional_cuda_devices(value: str | None) -> list[torch.device]:
    if value is None:
        return []
    return parse_cuda_devices(value)


def configure_sdpa_backend(backend: str) -> None:
    if backend == "auto":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)
        return
    if backend == "efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        return
    if backend == "flash_math":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        return
    if backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        return
    raise ValueError(f"Unsupported SDPA backend policy: {backend}")


def parse_sdpa_backends(value: str) -> list[str]:
    backends = [part.strip() for part in value.split(",") if part.strip()]
    if not backends:
        raise ValueError("--auto-plan-sdpa-backends must contain at least one backend")
    allowed = {"auto", "efficient", "flash_math", "math"}
    unknown = sorted(set(backends) - allowed)
    if unknown:
        raise ValueError(f"Unsupported SDPA backends in --auto-plan-sdpa-backends: {unknown}")
    return backends


def apply_auto_plan_runtime_defaults(args: argparse.Namespace) -> None:
    if not args.auto_plan:
        return
    args.compare_mode = "pipeline-memory-parallel"
    args.capacity_mode = "pipeline-memory-parallel"
    args.input_device = "cpu"
    args.cache_device = "cpu"
    args.offload_outputs_to_cpu = True
    if args.patch_embed_chunk_size is None:
        args.patch_embed_chunk_size = 32
    if args.head_device_index is None:
        args.head_device_index = 0
    if args.sdpa_backend == "auto":
        args.sdpa_backend = parse_sdpa_backends(args.auto_plan_sdpa_backends)[0]


def sorted_image_paths(image_dir: pathlib.Path, limit_frames: int | None) -> list[pathlib.Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes)
    if limit_frames is not None:
        paths = paths[:limit_frames]
    if not paths:
        raise RuntimeError(f"No image inputs found in {image_dir}")
    return paths


def load_images(args: argparse.Namespace, primary_device: torch.device) -> torch.Tensor:
    if args.image_dir is None:
        images = torch.rand(
            args.synthetic_frames,
            3,
            args.synthetic_height,
            args.synthetic_width,
            dtype=torch.float32,
        )
    else:
        image_paths = sorted_image_paths(args.image_dir, args.limit_frames)
        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=args.preprocess_mode,
            image_resolution=args.image_resolution,
        )
    if args.input_device == "cpu":
        return images
    return images.to(device=primary_device, non_blocking=True)


def load_model(args: argparse.Namespace, primary_device: torch.device) -> VGGTOmega:
    model = VGGTOmega().eval()
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise RuntimeError(f"Checkpoint not found: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    return model.to(device=primary_device)


def reset_memory_stats(devices: list[torch.device]) -> None:
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)


def synchronize(devices: list[torch.device]) -> None:
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)


def memory_summary(devices: list[torch.device]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for device in devices:
        summary[str(device)] = {
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
        }
    return summary


def tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def add_bytes(summary: dict[str, dict[str, int]], class_name: str, device: torch.device, byte_count: int) -> None:
    class_summary = summary.setdefault(class_name, {})
    device_name = str(device)
    class_summary[device_name] = class_summary.get(device_name, 0) + byte_count


def block_stage_index(block_idx: int, stage_splits: list[int]) -> int:
    return sum(split_block <= block_idx for split_block in stage_splits)


def parameter_memory_classes(model: VGGTOmega, stage_splits: list[int]) -> dict[str, dict[str, float]]:
    class_bytes: dict[str, dict[str, int]] = {}
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, value in tensors:
        if value.device.type == "meta":
            continue
        byte_count = tensor_bytes(value)
        if name.startswith("aggregator.patch_embed"):
            class_name = "input_patch_embed_weights"
        elif name.startswith("aggregator.frame_blocks.") or name.startswith("aggregator.inter_frame_blocks."):
            parts = name.split(".")
            block_idx = int(parts[2])
            class_name = f"aggregator_stage_{block_stage_index(block_idx, stage_splits)}_weights"
        elif name.startswith("aggregator."):
            class_name = "aggregator_token_and_buffer_state"
        elif name.startswith("camera_head.") or name.startswith("dense_head.") or name.startswith("text_alignment_head."):
            class_name = "head_weights"
        else:
            class_name = "other_weights"
        add_bytes(class_bytes, class_name, value.device, byte_count)

    return {
        class_name: {device: byte_count / (1024**3) for device, byte_count in devices.items()}
        for class_name, devices in sorted(class_bytes.items())
    }


def estimate_cached_aggregator_output_gb(model: VGGTOmega, images: torch.Tensor) -> float:
    if images.dim() == 4:
        num_frames, _, height, width = images.shape
        batch_size = 1
    elif images.dim() == 5:
        batch_size, num_frames, _, height, width = images.shape
    else:
        return 0.0

    patch_h = height // model.aggregator.patch_size
    patch_w = width // model.aggregator.patch_size
    num_tokens = model.aggregator.patch_token_start + patch_h * patch_w
    embed_dim = model.aggregator.camera_token.shape[-1]
    cached_layers = len(model.aggregator.cached_layer_indices)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    element_size = torch.empty((), dtype=amp_dtype).element_size()
    cached_bytes = batch_size * num_frames * num_tokens * (2 * embed_dim) * cached_layers * element_size
    return cached_bytes / (1024**3)


def estimate_output_gb(images: torch.Tensor) -> dict[str, float]:
    if images.dim() == 4:
        num_frames, num_channels, height, width = images.shape
        batch_size = 1
    elif images.dim() == 5:
        batch_size, num_frames, num_channels, height, width = images.shape
    else:
        return {}

    fp32_size = torch.empty((), dtype=torch.float32).element_size()
    return {
        "images": batch_size * num_frames * num_channels * height * width * fp32_size / (1024**3),
        "depth": batch_size * num_frames * height * width * fp32_size / (1024**3),
        "depth_conf": batch_size * num_frames * height * width * fp32_size / (1024**3),
        "pose_enc": batch_size * num_frames * 9 * fp32_size / (1024**3),
        "camera_and_register_tokens": batch_size * num_frames * 17 * 2048 * fp32_size / (1024**3),
    }


def image_shape_info(images: torch.Tensor) -> dict[str, int]:
    if images.dim() == 4:
        num_frames, num_channels, height, width = images.shape
        batch_size = 1
    elif images.dim() == 5:
        batch_size, num_frames, num_channels, height, width = images.shape
    else:
        raise ValueError(f"Unsupported image tensor shape for planning: {tuple(images.shape)}")
    return {
        "batch_size": int(batch_size),
        "num_frames": int(num_frames),
        "num_channels": int(num_channels),
        "height": int(height),
        "width": int(width),
    }


def token_estimate(model: VGGTOmega, images: torch.Tensor) -> dict[str, int | float | str]:
    shape = image_shape_info(images)
    patch_h = shape["height"] // model.aggregator.patch_size
    patch_w = shape["width"] // model.aggregator.patch_size
    frame_tokens = model.aggregator.patch_token_start + patch_h * patch_w
    sequence_tokens = shape["num_frames"] * frame_tokens
    embed_dim = int(model.aggregator.camera_token.shape[-1])
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    live_token_gb = shape["batch_size"] * sequence_tokens * embed_dim * torch.empty((), dtype=amp_dtype).element_size()
    return {
        **shape,
        "patch_grid_h": patch_h,
        "patch_grid_w": patch_w,
        "frame_tokens": frame_tokens,
        "sequence_tokens": sequence_tokens,
        "embed_dim": embed_dim,
        "amp_dtype": str(amp_dtype),
        "live_token_gb": live_token_gb / (1024**3),
    }


def device_inventory(devices: list[torch.device]) -> list[dict[str, Any]]:
    inventory = []
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        inventory.append(
            {
                "device": str(device),
                "name": props.name,
                "total_gb": total_bytes / (1024**3),
                "free_gb": free_bytes / (1024**3),
                "used_gb": (total_bytes - free_bytes) / (1024**3),
                "multi_processor_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return inventory


def block_weight_gb(model: VGGTOmega, block_idx: int) -> float:
    prefixes = (
        f"aggregator.frame_blocks.{block_idx}.",
        f"aggregator.inter_frame_blocks.{block_idx}.",
    )
    byte_count = 0
    for name, value in model.named_parameters():
        if name.startswith(prefixes):
            byte_count += tensor_bytes(value)
    for name, value in model.named_buffers():
        if name.startswith(prefixes):
            byte_count += tensor_bytes(value)
    return byte_count / (1024**3)


def choose_weighted_stage_splits(
    model: VGGTOmega,
    stage_devices: list[torch.device],
    inventory: list[dict[str, Any]],
    reserve_primary_for_patch: bool,
) -> list[int]:
    depth = model.aggregator.depth
    if len(stage_devices) == 1:
        return []

    active_stage_devices = stage_devices[1:] if reserve_primary_for_patch else stage_devices
    if not active_stage_devices:
        return []

    device_free = {entry["device"]: max(float(entry["free_gb"]), 1.0) for entry in inventory}
    capacities = [device_free[str(device)] for device in active_stage_devices]
    total_capacity = sum(capacities)
    block_weights = [block_weight_gb(model, block_idx) for block_idx in range(depth)]
    total_weight = sum(block_weights)
    splits = [0] if reserve_primary_for_patch else []
    cumulative = 0.0
    next_target_index = 0
    targets = []
    running_capacity = 0.0
    for capacity in capacities[:-1]:
        running_capacity += capacity
        targets.append(total_weight * running_capacity / total_capacity)

    for block_idx, block_weight in enumerate(block_weights, start=1):
        cumulative += block_weight
        if next_target_index < len(targets) and cumulative >= targets[next_target_index]:
            split_block = min(max(block_idx, 0), depth)
            if not splits or split_block > splits[-1]:
                splits.append(split_block)
            next_target_index += 1

    while len(splits) < len(stage_devices) - 1:
        candidate = 0 if not splits else min(depth, splits[-1] + 1)
        splits.append(candidate)
    return splits[: len(stage_devices) - 1]


def choose_query_devices(
    devices: list[torch.device],
    stage_splits: list[int],
    inventory: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[torch.device]:
    if args.auto_plan_query_shards == "off" or len(devices) < 3:
        return []
    if args.query_blockwise_devices:
        return parse_optional_cuda_devices(args.query_blockwise_devices)

    stage_block_counts = {str(device): 0 for device in devices}
    for block_idx in range(24):
        stage_idx = min(block_stage_index(block_idx, stage_splits), len(devices) - 1)
        stage_block_counts[str(devices[stage_idx])] += 1
    free_by_device = {entry["device"]: float(entry["free_gb"]) for entry in inventory}
    eligible_devices = [device for device in devices if stage_block_counts[str(device)] == 0]
    if len(eligible_devices) < 2:
        return []
    ranked = sorted(
        eligible_devices,
        key=lambda device: (stage_block_counts[str(device)], -free_by_device[str(device)]),
    )
    return ranked[: min(2, len(ranked))]


def auto_pipeline_plan(
    args: argparse.Namespace,
    model: VGGTOmega,
    images: torch.Tensor,
    devices: list[torch.device],
) -> dict[str, Any]:
    inventory = device_inventory(devices)
    stage_devices = devices
    reserve_primary_for_patch = len(stage_devices) > 1
    stage_splits = choose_weighted_stage_splits(
        model,
        stage_devices,
        inventory,
        reserve_primary_for_patch=reserve_primary_for_patch,
    )
    stage_block_counts = {str(device): 0 for device in stage_devices}
    for block_idx in range(model.aggregator.depth):
        stage_idx = min(block_stage_index(block_idx, stage_splits), len(stage_devices) - 1)
        stage_block_counts[str(stage_devices[stage_idx])] += 1
    query_devices = choose_query_devices(devices, stage_splits, inventory, args)
    if query_devices and args.query_block_size == 2048:
        args.query_block_size = 8192

    args.stage_splits = ",".join(str(split) for split in stage_splits)
    args.cache_device = "cpu"
    args.head_device_index = 0
    args.input_device = "cpu"
    args.offload_outputs_to_cpu = True
    args.query_blockwise_devices = ",".join(str(device.index) for device in query_devices) if query_devices else None

    return {
        "source": "auto",
        "device_inventory": inventory,
        "token_estimate": token_estimate(model, images),
        "roles": {
            "patch_device": str(devices[0]),
            "head_device": str(devices[args.head_device_index]),
            "cache_device": args.cache_device,
            "output_device": "cpu",
            "stage_devices": [str(device) for device in stage_devices],
            "query_worker_devices": [str(device) for device in query_devices],
        },
        "stage_splits": stage_splits,
        "stage_block_counts": stage_block_counts,
        "query_block_size": args.query_block_size,
        "sdpa_backend": args.sdpa_backend,
        "rationale": (
            "Auto-plan keeps inputs/cached outputs/final outputs on CPU, reserves the first CUDA device "
            "for patch embedding and heads when multiple GPUs are visible, splits aggregator blocks across "
            "remaining stage capacity, and only assigns query-worker roles to the lowest estimated stage-pressure devices."
        ),
    }


def placement_summary(
    model: VGGTOmega,
    images: torch.Tensor,
    devices: list[torch.device],
    stage_splits: list[int],
    stage_devices: list[str],
    cache_device: str,
    head_device: str,
    query_blockwise_devices: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "devices": [str(device) for device in devices],
        "stage_devices": stage_devices,
        "stage_splits": stage_splits,
        "cache_device": cache_device,
        "head_device": head_device,
        "input_device": args.input_device,
        "patch_embed_chunk_size": args.patch_embed_chunk_size,
        "offload_outputs_to_cpu": bool(args.offload_outputs_to_cpu),
        "query_blockwise_devices": query_blockwise_devices,
        "query_block_size": args.query_block_size,
        "sdpa_backend": args.sdpa_backend,
        "parameter_memory_gb_by_class": parameter_memory_classes(model, stage_splits),
        "estimated_cached_aggregator_outputs_gb": estimate_cached_aggregator_output_gb(model, images),
        "estimated_output_tensors_gb": estimate_output_gb(images),
    }


def tensor_shapes(outputs: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            shapes[key] = list(value.shape)
    return shapes


def tensor_finite_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    summary = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
    }
    if finite_values.numel():
        finite_float = finite_values.float()
        summary["finite_min"] = float(finite_float.min().item())
        summary["finite_max"] = float(finite_float.max().item())
    else:
        summary["finite_min"] = None
        summary["finite_max"] = None
    return summary


def output_finite_summary(outputs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in OUTPUT_KEYS:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            summary[key] = tensor_finite_summary(value)
    return summary


def run_single(
    args: argparse.Namespace,
    images: torch.Tensor,
    devices: list[torch.device],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_device = devices[0]
    model = load_model(args, primary_device)
    reset_memory_stats(devices)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(images)
    synchronize(devices)
    elapsed = time.perf_counter() - start
    return outputs, {
        "mode": "single",
        "frame_count": int(images.shape[0]),
        "image_shape": list(images.shape),
        "elapsed_sec": elapsed,
        "memory": memory_summary(devices),
        "output_shapes": tensor_shapes(outputs),
        "finite": output_finite_summary(outputs),
    }


def run_head_parallel(
    args: argparse.Namespace,
    images: torch.Tensor,
    devices: list[torch.device],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_device = devices[0]
    model = load_model(args, primary_device)
    model.enable_head_parallelism(devices)
    reset_memory_stats(devices)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(images)
    synchronize(devices)
    elapsed = time.perf_counter() - start
    return outputs, {
        "mode": "head-parallel",
        "status": "completed",
        "finding": (
            "Aggregator inter-frame blocks and the camera-head trunk split already-projected "
            "Q/K/V tensors by attention-head range, run SDPA shards on the requested CUDA "
            "devices, then gather heads before the native output projection on the primary device."
        ),
        "frame_count": int(images.shape[0]),
        "image_shape": list(images.shape),
        "devices": [str(device) for device in devices],
        "elapsed_sec": elapsed,
        "memory": memory_summary(devices),
        "output_shapes": tensor_shapes(outputs),
        "finite": output_finite_summary(outputs),
    }


def run_memory_parallel(
    args: argparse.Namespace,
    images: torch.Tensor,
    devices: list[torch.device],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_device = devices[0]
    model = load_model(args, primary_device)
    model.enable_memory_parallelism(devices)
    reset_memory_stats(devices)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(images)
    synchronize(devices)
    elapsed = time.perf_counter() - start
    return outputs, {
        "mode": "memory-parallel",
        "status": "completed",
        "finding": (
            "Aggregator execution stays on the primary CUDA device, while cached aggregator "
            "layer outputs are offloaded to the secondary CUDA device and camera/dense heads "
            "run there to reduce retained primary-device memory."
        ),
        "frame_count": int(images.shape[0]),
        "image_shape": list(images.shape),
        "devices": [str(device) for device in devices],
        "cache_device": str(devices[1]),
        "elapsed_sec": elapsed,
        "memory": memory_summary(devices),
        "output_shapes": tensor_shapes(outputs),
        "finite": output_finite_summary(outputs),
    }


def run_balanced_memory_parallel(
    args: argparse.Namespace,
    images: torch.Tensor,
    devices: list[torch.device],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_device = devices[0]
    model = load_model(args, primary_device)
    model.enable_balanced_memory_parallelism(devices, split_block=args.split_block)
    reset_memory_stats(devices)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(images)
    synchronize(devices)
    elapsed = time.perf_counter() - start
    return outputs, {
        "mode": "balanced-memory-parallel",
        "status": "completed",
        "finding": (
            "Patch embedding and early aggregator blocks run on the primary CUDA device, "
            "later aggregator blocks and live tokens move to the secondary CUDA device, "
            "and cached outputs plus camera/dense heads remain on the secondary device."
        ),
        "frame_count": int(images.shape[0]),
        "image_shape": list(images.shape),
        "devices": [str(device) for device in devices],
        "split_block": args.split_block,
        "stage_device": str(devices[1]),
        "elapsed_sec": elapsed,
        "memory": memory_summary(devices),
        "output_shapes": tensor_shapes(outputs),
        "finite": output_finite_summary(outputs),
    }


def run_pipeline_memory_parallel(
    args: argparse.Namespace,
    images: torch.Tensor,
    devices: list[torch.device],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_device = devices[0]
    model = load_model(args, primary_device)
    planner_summary = auto_pipeline_plan(args, model, images, devices) if args.auto_plan else {"source": "manual"}
    stage_splits = parse_stage_splits(args.stage_splits)
    model.set_patch_embed_chunk_size(args.patch_embed_chunk_size)
    if args.offload_outputs_to_cpu:
        model.set_output_device("cpu")
    cache_device_index = None if args.cache_device is not None else args.cache_device_index
    model.enable_pipeline_memory_parallelism(
        devices,
        stage_splits=stage_splits,
        cache_device_index=cache_device_index,
        cache_device=args.cache_device,
        head_device_index=args.head_device_index,
    )
    query_blockwise_devices = parse_optional_cuda_devices(args.query_blockwise_devices)
    if query_blockwise_devices:
        model.enable_global_inter_frame_query_blockwise_parallelism(
            query_blockwise_devices,
            query_block_size=args.query_block_size,
        )
    if args.cache_device is None:
        cache_device = str(devices[args.cache_device_index])
        stage_devices = [str(device) for index, device in enumerate(devices) if index != args.cache_device_index]
        head_device = str(devices[args.head_device_index]) if args.head_device_index is not None else cache_device
    else:
        cache_device = args.cache_device
        stage_devices = [str(device) for device in devices]
        head_device = str(devices[args.head_device_index]) if args.head_device_index is not None else str(devices[-1])
    placement = placement_summary(
        model,
        images,
        devices,
        stage_splits,
        stage_devices,
        cache_device,
        head_device,
        [str(device) for device in query_blockwise_devices],
        args,
    )
    summary = {
        "mode": "pipeline-memory-parallel",
        "status": "completed",
        "finding": (
            "Aggregator blocks run across ordered stage CUDA devices with live tokens moved at "
            "the requested split blocks; cached outputs and camera/dense heads run on the "
            "configured cache CUDA device."
        ),
        "frame_count": int(images.shape[0]),
        "image_shape": list(images.shape),
        "devices": [str(device) for device in devices],
        "stage_devices": stage_devices,
        "stage_splits": stage_splits,
        "cache_device": cache_device,
        "head_device": head_device,
        "patch_embed_chunk_size": args.patch_embed_chunk_size,
        "input_device": args.input_device,
        "offload_outputs_to_cpu": bool(args.offload_outputs_to_cpu),
        "query_blockwise_devices": [str(device) for device in query_blockwise_devices],
        "query_block_size": args.query_block_size,
        "sdpa_backend": args.sdpa_backend,
        "planner": planner_summary,
        "placement": placement,
    }
    reset_memory_stats(devices)
    start = time.perf_counter()
    with torch.inference_mode():
        try:
            outputs = model(images)
        except torch.cuda.OutOfMemoryError as exc:
            summary["status"] = "oom"
            summary["memory"] = memory_summary(devices)
            raise ProfileOutOfMemoryError(str(exc), summary) from exc
    synchronize(devices)
    elapsed = time.perf_counter() - start
    summary["elapsed_sec"] = elapsed
    summary["memory"] = memory_summary(devices)
    summary["output_shapes"] = tensor_shapes(outputs)
    summary["finite"] = output_finite_summary(outputs)
    return outputs, summary


def run_profile_mode(
    mode: str,
    args: argparse.Namespace,
    images: torch.Tensor,
    devices: list[torch.device],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode == "single":
        return run_single(args, images, devices)
    if mode == "head-parallel":
        return run_head_parallel(args, images, devices)
    if mode == "memory-parallel":
        return run_memory_parallel(args, images, devices)
    if mode == "balanced-memory-parallel":
        return run_balanced_memory_parallel(args, images, devices)
    if mode == "pipeline-memory-parallel":
        return run_pipeline_memory_parallel(args, images, devices)
    raise AssertionError(f"Unhandled profile mode: {mode}")


def run_plan(args: argparse.Namespace, devices: list[torch.device]) -> dict[str, Any]:
    primary_device = devices[0]
    images = load_images(args, primary_device)
    model = load_model(args, primary_device)
    planner_summary = auto_pipeline_plan(args, model, images, devices) if args.auto_plan else {"source": "manual"}
    stage_splits = parse_stage_splits(args.stage_splits)
    return {
        "mode": "plan",
        "status": "completed",
        "devices": [str(device) for device in devices],
        "frame_count": int(images.shape[0]),
        "image_shape": list(images.shape),
        "stage_splits": stage_splits,
        "cache_device": args.cache_device,
        "head_device_index": args.head_device_index,
        "input_device": args.input_device,
        "offload_outputs_to_cpu": bool(args.offload_outputs_to_cpu),
        "query_blockwise_devices": [str(device) for device in parse_optional_cuda_devices(args.query_blockwise_devices)],
        "query_block_size": args.query_block_size,
        "sdpa_backend": args.sdpa_backend,
        "planner": planner_summary,
    }


def run_fsdp(args: argparse.Namespace, devices: list[torch.device]) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        return {
            "mode": "fsdp",
            "status": "not_run",
            "finding": (
                "FSDP is a multi-process sharding strategy. Launch this mode with "
                "`torchrun --nproc_per_node=N` to measure parameter sharding; a "
                "single Python process cannot use FSDP as the requested multi-GPU path."
            ),
        }

    import torch.distributed
    from torch.distributed.fsdp import FullyShardedDataParallel

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl")
    images = load_images(args, device)
    model = load_model(args, device)
    model = FullyShardedDataParallel(model, device_id=device)
    reset_memory_stats([device])
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(images)
    synchronize([device])
    elapsed = time.perf_counter() - start
    result = {
        "mode": "fsdp",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "frame_count": int(images.shape[0]),
        "elapsed_sec": elapsed,
        "memory": memory_summary([device]),
        "output_shapes": tensor_shapes(outputs),
        "finite": output_finite_summary(outputs),
    }
    torch.distributed.destroy_process_group()
    return result


def cpu_outputs(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for key in OUTPUT_KEYS:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            values[key] = value.detach().float().cpu()
    return values


def compare_outputs(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for key in OUTPUT_KEYS:
        if key not in reference or key not in candidate:
            comparison[key] = {"status": "missing"}
            continue
        ref = reference[key]
        cand = candidate[key]
        if ref.shape != cand.shape:
            comparison[key] = {
                "status": "shape_mismatch",
                "reference_shape": list(ref.shape),
                "candidate_shape": list(cand.shape),
            }
            continue

        ref_summary = tensor_finite_summary(ref)
        cand_summary = tensor_finite_summary(cand)
        if (
            ref_summary["nan_count"]
            or ref_summary["inf_count"]
            or cand_summary["nan_count"]
            or cand_summary["inf_count"]
        ):
            comparison[key] = {
                "status": "failed_nonfinite",
                "reference": ref_summary,
                "candidate": cand_summary,
            }
            continue

        diff = (ref - cand).abs()
        comparison[key] = {
            "status": "passed" if torch.allclose(ref, cand, atol=atol, rtol=rtol) else "failed",
            "shape": list(ref.shape),
            "reference": ref_summary,
            "candidate": cand_summary,
            "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        }
    return comparison


def parity_passed(parity: dict[str, Any]) -> bool:
    return all(value.get("status") == "passed" for value in parity.values())


def run_compare(args: argparse.Namespace, images: torch.Tensor, devices: list[torch.device]) -> dict[str, Any]:
    configure_sdpa_backend(args.sdpa_backend)
    single_outputs, single_summary = run_single(args, images, devices)
    reference = cpu_outputs(single_outputs)
    del single_outputs
    torch.cuda.empty_cache()
    parallel_outputs, parallel_summary = run_profile_mode(args.compare_mode, args, images, devices)
    candidate = cpu_outputs(parallel_outputs)
    parity = compare_outputs(reference, candidate, args.parity_atol, args.parity_rtol)
    return {
        "mode": "compare",
        "compare_mode": args.compare_mode,
        "status": "passed" if parity_passed(parity) else "failed",
        "single": single_summary,
        args.compare_mode.replace("-", "_"): parallel_summary,
        "parity": parity,
    }


def compact_probe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    candidate_key = summary["compare_mode"].replace("-", "_")
    candidate = summary[candidate_key]
    return {
        "status": summary["status"],
        "frame_count": candidate["frame_count"],
        "image_shape": candidate["image_shape"],
        "sdpa_backend": candidate.get("sdpa_backend"),
        "stage_splits": candidate.get("stage_splits"),
        "query_blockwise_devices": candidate.get("query_blockwise_devices"),
        "query_block_size": candidate.get("query_block_size"),
        "single_memory": summary["single"].get("memory"),
        "candidate_memory": candidate.get("memory"),
        "parity": summary["parity"],
    }


def append_arg(command: list[str], name: str, value: str | int | float | pathlib.Path | None) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def run_compare_probe_subprocess(
    args: argparse.Namespace,
    devices: list[torch.device],
    backend: str,
    frame_count: int,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="vggt_omega_probe_", suffix=".json", delete=False) as handle:
        output_path = pathlib.Path(handle.name)

    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--mode",
        "compare",
        "--auto-plan",
        "--compare-mode",
        "pipeline-memory-parallel",
        "--devices",
        ",".join(str(device.index) for device in devices),
        "--sdpa-backend",
        backend,
        "--limit-frames",
        str(frame_count),
        "--synthetic-frames",
        str(frame_count),
        "--synthetic-height",
        str(args.synthetic_height),
        "--synthetic-width",
        str(args.synthetic_width),
        "--image-resolution",
        str(args.image_resolution),
        "--preprocess-mode",
        args.preprocess_mode,
        "--parity-atol",
        str(args.parity_atol),
        "--parity-rtol",
        str(args.parity_rtol),
        "--output-json",
        str(output_path),
    ]
    append_arg(command, "--checkpoint", args.checkpoint)
    append_arg(command, "--image-dir", args.image_dir)
    append_arg(command, "--patch-embed-chunk-size", args.patch_embed_chunk_size)
    if args.auto_plan_query_shards == "off":
        command.append("--auto-plan-query-shards")
        command.append("off")
    if args.query_blockwise_devices:
        command.extend(["--query-blockwise-devices", args.query_blockwise_devices])
    command.extend(["--query-block-size", str(args.query_block_size)])

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        if completed.returncode != 0:
            return {
                "status": "exception",
                "frame_count": frame_count,
                "error_type": "SubprocessError",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-4000:],
                "stdout_tail": completed.stdout[-2000:],
            }
        summary = json.loads(output_path.read_text(encoding="utf-8"))
        return compact_probe_summary(summary)
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def run_compare_probe_in_process(
    args: argparse.Namespace,
    devices: list[torch.device],
    backend: str,
    frame_count: int,
) -> dict[str, Any]:
    probe_args = copy.copy(args)
    probe_args.mode = "compare"
    probe_args.compare_mode = "pipeline-memory-parallel"
    probe_args.sdpa_backend = backend
    probe_args.limit_frames = frame_count
    probe_args.synthetic_frames = frame_count
    images = load_images(probe_args, devices[0])
    summary = run_compare(probe_args, images, devices)
    del images
    return compact_probe_summary(summary)


def run_auto_plan_parity_probes(args: argparse.Namespace, devices: list[torch.device]) -> dict[str, Any]:
    frames = parse_frame_counts(args.auto_plan_parity_frames)
    backend_results = []
    for backend in parse_sdpa_backends(args.auto_plan_sdpa_backends):
        frame_results = []
        backend_passed = True
        for frame_count in frames:
            try:
                if args.auto_plan_probe_isolation == "subprocess":
                    probe_summary = run_compare_probe_subprocess(args, devices, backend, frame_count)
                else:
                    probe_summary = run_compare_probe_in_process(args, devices, backend, frame_count)
                frame_results.append(probe_summary)
                backend_passed = backend_passed and probe_summary["status"] == "passed"
            except Exception as exc:
                frame_results.append(
                    {
                        "status": "exception",
                        "frame_count": frame_count,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                backend_passed = False
            finally:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if not backend_passed:
                break
        backend_results.append({"backend": backend, "status": "passed" if backend_passed else "failed", "frames": frame_results})
        if backend_passed:
            args.sdpa_backend = backend
            configure_sdpa_backend(backend)
            return {
                "status": "passed",
                "selected_backend": backend,
                "requested_frames": frames,
                "backend_results": backend_results,
            }
    return {
        "status": "failed",
        "selected_backend": None,
        "requested_frames": frames,
        "backend_results": backend_results,
    }


def run_capacity(args: argparse.Namespace, devices: list[torch.device]) -> dict[str, Any]:
    parity_probes = None
    if args.auto_plan and not args.skip_auto_plan_parity:
        parity_probes = run_auto_plan_parity_probes(args, devices)
        if parity_probes["status"] != "passed":
            return {
                "mode": "capacity",
                "capacity_mode": args.capacity_mode,
                "status": "failed_parity_probe",
                "auto_plan_parity": parity_probes,
                "frame_counts": parse_frame_counts(args.frame_counts),
                "results": [],
            }

    results = []
    frame_counts = parse_frame_counts(args.frame_counts)
    for frame_count in frame_counts:
        args.limit_frames = frame_count
        args.synthetic_frames = frame_count
        images = None
        try:
            images = load_images(args, devices[0])
            if int(images.shape[0]) != frame_count:
                results.append(
                    {
                        "frame_count": frame_count,
                        "status": "skipped",
                        "reason": f"requested {frame_count} frames but loaded {int(images.shape[0])}",
                    }
                )
                continue
            _outputs, summary = run_profile_mode(args.capacity_mode, args, images, devices)
            summary["status"] = "completed"
            results.append(summary)
            del _outputs
        except ProfileOutOfMemoryError as exc:
            summary = dict(exc.summary)
            summary["frame_count"] = frame_count
            summary["status"] = "oom"
            summary["error"] = str(exc)
            results.append(summary)
            break
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"frame_count": frame_count, "status": "oom", "error": str(exc)})
            break
        except Exception as exc:
            results.append(
                {
                    "frame_count": frame_count,
                    "status": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "memory": memory_summary(devices),
                }
            )
            break
        finally:
            if images is not None:
                del images
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return {
        "mode": "capacity",
        "capacity_mode": args.capacity_mode,
        "auto_plan_parity": parity_probes,
        "frame_counts": frame_counts,
        "results": results,
    }


def output_path_for_rank(output_json: pathlib.Path | None) -> pathlib.Path | None:
    if output_json is None:
        return None
    rank = os.environ.get("RANK")
    if rank is None:
        return output_json
    return output_json.with_name(f"{output_json.stem}.rank{rank}{output_json.suffix}")


def emit_summary(summary: dict[str, Any], output_json: pathlib.Path | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    resolved_output = output_path_for_rank(output_json)
    if resolved_output is not None:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VGGT-Omega multi-device profiling.")
    apply_auto_plan_runtime_defaults(args)
    configure_sdpa_backend(args.sdpa_backend)
    devices = parse_cuda_devices(args.devices)

    if args.mode == "plan":
        emit_summary(run_plan(args, devices), args.output_json)
        return

    if args.mode == "fsdp":
        emit_summary(run_fsdp(args, devices), args.output_json)
        return

    if args.mode == "capacity":
        emit_summary(run_capacity(args, devices), args.output_json)
        return

    images = load_images(args, devices[0])
    if args.mode in {
        "single",
        "head-parallel",
        "memory-parallel",
        "balanced-memory-parallel",
        "pipeline-memory-parallel",
    }:
        _outputs, summary = run_profile_mode(args.mode, args, images, devices)
    elif args.mode == "compare":
        summary = run_compare(args, images, devices)
    else:
        raise AssertionError(f"Unhandled mode: {args.mode}")
    emit_summary(summary, args.output_json)


if __name__ == "__main__":
    main()
