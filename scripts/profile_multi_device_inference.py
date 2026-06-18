#!/usr/bin/env python
"""Profile experimental multi-device VGGT-Omega inference paths."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VGGT-Omega single-GPU and experimental GPU-parallel inference "
            "for memory/parity investigation."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("single", "dataparallel", "fsdp", "head-parallel", "compare"),
        default="single",
        help="Inference path to profile. compare runs single and head-parallel sequentially.",
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


def tensor_shapes(outputs: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            shapes[key] = list(value.shape)
    return shapes


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
    }


def run_dataparallel(
    args: argparse.Namespace,
    images: torch.Tensor,
    devices: list[torch.device],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_device = devices[0]
    model = load_model(args, primary_device)
    device_ids = [int(device.index) for device in devices if device.index is not None]
    parallel_model = torch.nn.DataParallel(model, device_ids=device_ids)
    native_batch = images.unsqueeze(0)
    reset_memory_stats(devices)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = parallel_model(native_batch)
    synchronize(devices)
    elapsed = time.perf_counter() - start
    return outputs, {
        "mode": "dataparallel",
        "status": "completed",
        "finding": (
            "VGGT-Omega native video inference uses batch_size=1 and frames on axis 1; "
            "DataParallel splits axis 0, so this path does not shard frames for the usual input."
        ),
        "frame_count": int(images.shape[0]),
        "native_batch_shape": list(native_batch.shape),
        "elapsed_sec": elapsed,
        "memory": memory_summary(devices),
        "output_shapes": tensor_shapes(outputs),
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
            "Attention heads in aggregator inter-frame blocks and the camera-head trunk "
            "were split across the requested CUDA devices. QKV projection and output "
            "projection remain on the primary device in this prototype."
        ),
        "frame_count": int(images.shape[0]),
        "image_shape": list(images.shape),
        "devices": [str(device) for device in devices],
        "elapsed_sec": elapsed,
        "memory": memory_summary(devices),
        "output_shapes": tensor_shapes(outputs),
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
        "rank": int(os.environ["RANK"]),
        "local_rank": local_rank,
        "world_size": world_size,
        "frame_count": int(images.shape[0]),
        "elapsed_sec": elapsed,
        "memory": memory_summary([device]),
        "output_shapes": tensor_shapes(outputs),
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
        diff = (ref - cand).abs()
        comparison[key] = {
            "status": "passed" if torch.allclose(ref, cand, atol=atol, rtol=rtol) else "failed",
            "shape": list(ref.shape),
            "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        }
    return comparison


def emit_summary(summary: dict[str, Any], output_json: pathlib.Path | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VGGT-Omega multi-device profiling.")
    devices = parse_cuda_devices(args.devices)

    if args.mode == "fsdp":
        emit_summary(run_fsdp(args, devices), args.output_json)
        return

    images = load_images(args, devices[0])
    if args.mode == "single":
        _outputs, summary = run_single(args, images, devices)
    elif args.mode == "dataparallel":
        _outputs, summary = run_dataparallel(args, images, devices)
    elif args.mode == "head-parallel":
        _outputs, summary = run_head_parallel(args, images, devices)
    elif args.mode == "compare":
        single_outputs, single_summary = run_single(args, images, devices)
        reference = cpu_outputs(single_outputs)
        del single_outputs
        torch.cuda.empty_cache()
        head_outputs, head_summary = run_head_parallel(args, images, devices)
        candidate = cpu_outputs(head_outputs)
        summary = {
            "mode": "compare",
            "single": single_summary,
            "head_parallel": head_summary,
            "parity": compare_outputs(reference, candidate, args.parity_atol, args.parity_rtol),
        }
    else:
        raise AssertionError(f"Unhandled mode: {args.mode}")
    emit_summary(summary, args.output_json)


if __name__ == "__main__":
    main()
