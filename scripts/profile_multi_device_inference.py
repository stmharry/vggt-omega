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
        choices=("single", "head-parallel", "fsdp", "compare", "capacity"),
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
    parser.add_argument(
        "--capacity-mode",
        choices=("single", "head-parallel"),
        default="head-parallel",
        help="Inference path used by capacity mode.",
    )
    parser.add_argument(
        "--frame-counts",
        default="50,100,200,300,400",
        help="Comma-separated frame counts for capacity mode.",
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


def run_capacity(args: argparse.Namespace, devices: list[torch.device]) -> dict[str, Any]:
    results = []
    frame_counts = parse_frame_counts(args.frame_counts)
    for frame_count in frame_counts:
        args.limit_frames = frame_count
        args.synthetic_frames = frame_count
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
        try:
            if args.capacity_mode == "single":
                _outputs, summary = run_single(args, images, devices)
            elif args.capacity_mode == "head-parallel":
                _outputs, summary = run_head_parallel(args, images, devices)
            else:
                raise AssertionError(f"Unhandled capacity mode: {args.capacity_mode}")
            summary["status"] = "completed"
            results.append(summary)
            del _outputs
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"frame_count": frame_count, "status": "oom", "error": str(exc)})
            break
        finally:
            del images
            torch.cuda.empty_cache()
    return {
        "mode": "capacity",
        "capacity_mode": args.capacity_mode,
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
    devices = parse_cuda_devices(args.devices)

    if args.mode == "fsdp":
        emit_summary(run_fsdp(args, devices), args.output_json)
        return

    if args.mode == "capacity":
        emit_summary(run_capacity(args, devices), args.output_json)
        return

    images = load_images(args, devices[0])
    if args.mode == "single":
        _outputs, summary = run_single(args, images, devices)
    elif args.mode == "head-parallel":
        _outputs, summary = run_head_parallel(args, images, devices)
    elif args.mode == "compare":
        single_outputs, single_summary = run_single(args, images, devices)
        reference = cpu_outputs(single_outputs)
        del single_outputs
        torch.cuda.empty_cache()
        parallel_outputs, parallel_summary = run_head_parallel(args, images, devices)
        candidate = cpu_outputs(parallel_outputs)
        parity = compare_outputs(reference, candidate, args.parity_atol, args.parity_rtol)
        summary = {
            "mode": "compare",
            "status": "passed" if parity_passed(parity) else "failed",
            "single": single_summary,
            "head_parallel": parallel_summary,
            "parity": parity,
        }
    else:
        raise AssertionError(f"Unhandled mode: {args.mode}")
    emit_summary(summary, args.output_json)


if __name__ == "__main__":
    main()
