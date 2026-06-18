from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def normalize_map(values: torch.Tensor) -> torch.Tensor:
    minimum = values.amin()
    maximum = values.amax()
    return (values - minimum) / (maximum - minimum + 1e-6)


def otsu_threshold(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    hist, bin_edges = np.histogram(finite, bins=256, range=(float(finite.min()), float(finite.max())))
    total = hist.sum()
    if total == 0:
        return 1.0
    sum_total = np.dot(hist, bin_edges[:-1])
    weight_background = 0.0
    sum_background = 0.0
    best_variance = -1.0
    best_threshold = float(finite.mean())
    for index, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += count * bin_edges[index]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = float(bin_edges[index])
    return best_threshold


def layer_band(layer_count: int, start_fraction: float, end_fraction: float) -> slice:
    start = max(0, min(layer_count - 1, int(round(layer_count * start_fraction))))
    end = max(start + 1, min(layer_count, int(round(layer_count * end_fraction))))
    return slice(start, end)


def temporal_similarity(
    reference: torch.Tensor,
    sources: torch.Tensor,
) -> torch.Tensor:
    reference = torch.nn.functional.normalize(reference.float(), dim=-1)
    sources = torch.nn.functional.normalize(sources.float(), dim=-1)
    return (reference.unsqueeze(0) * sources).sum(dim=-1).mean(dim=(0, 1, 2))


def temporal_variance(
    reference: torch.Tensor,
    sources: torch.Tensor,
) -> torch.Tensor:
    reference = torch.nn.functional.normalize(reference.float(), dim=-1)
    sources = torch.nn.functional.normalize(sources.float(), dim=-1)
    similarities = (reference.unsqueeze(0) * sources).sum(dim=-1)
    return similarities.std(dim=0).mean(dim=(0, 1))


def patch_saliency_maps(
    motion_features: dict[str, Any],
    temporal_offsets: tuple[int, ...] = (-6, -4, -2, 2, 4, 6),
) -> torch.Tensor:
    if "global_tok_q" not in motion_features or "global_tok_k" not in motion_features:
        raise ValueError("motion_features must include global_tok_q and global_tok_k")
    global_q = motion_features["global_tok_q"]
    global_k = motion_features["global_tok_k"]
    if global_q.ndim != 6 or global_k.shape != global_q.shape:
        raise ValueError(
            "Expected global_tok_q/global_tok_k with shape "
            "(layers, batch, frames, heads, patches, head_dim)"
        )
    if global_q.shape[1] != 1:
        raise ValueError("Motion mask extraction currently expects batch size 1")

    q = global_q[:, 0]
    k = global_k[:, 0]
    layer_count, frame_count = q.shape[:2]
    early = layer_band(layer_count, 0.0, 0.35)
    middle = layer_band(layer_count, 0.35, 0.75)
    deep = layer_band(layer_count, 0.75, 1.0)
    maps = []
    offsets = torch.tensor(temporal_offsets, dtype=torch.long)

    for frame_index in range(frame_count):
        source_indices = frame_index + offsets
        source_indices = source_indices[(source_indices >= 0) & (source_indices < frame_count)]
        if len(source_indices) == 0:
            maps.append(torch.zeros(q.shape[-2], dtype=torch.float32))
            continue

        early_similarity = temporal_similarity(q[early, frame_index], q[early][:, source_indices])
        deep_similarity = temporal_similarity(k[deep, frame_index], k[deep][:, source_indices])
        middle_variance = temporal_variance(q[middle, frame_index], q[middle][:, source_indices])
        saliency = (1.0 - early_similarity.clamp(-1, 1)) * (1.0 - deep_similarity.clamp(-1, 1))
        saliency = saliency * normalize_map(middle_variance)
        maps.append(normalize_map(saliency).cpu())

    return torch.stack(maps, dim=0)


def resize_saliency_maps(
    patch_maps: torch.Tensor,
    patch_grid_size: tuple[int, int],
    image_shape_hw: tuple[int, int],
) -> np.ndarray:
    frame_count = patch_maps.shape[0]
    patch_height, patch_width = patch_grid_size
    image_height, image_width = image_shape_hw
    saliency_maps = []
    for frame_index in range(frame_count):
        patch_map = patch_maps[frame_index].reshape(patch_height, patch_width).numpy()
        saliency = cv2.resize(
            patch_map,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency_maps.append(saliency.astype(np.float32, copy=False))
    return np.stack(saliency_maps, axis=0)


def masks_from_saliency(saliency_maps: np.ndarray, threshold_mode: str = "otsu") -> np.ndarray:
    masks = []
    for saliency in saliency_maps:
        if threshold_mode == "otsu":
            threshold = otsu_threshold(saliency)
        elif threshold_mode == "mean":
            threshold = float(np.nanmean(saliency))
        else:
            raise ValueError("threshold_mode must be 'otsu' or 'mean'")
        mask = saliency >= threshold
        mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        masks.append(mask > 0)
    return np.stack(masks, axis=0)


def dynamic_masks_from_motion_features(
    motion_features: dict[str, Any],
    image_shape_hw: tuple[int, int],
    threshold_mode: str = "otsu",
) -> tuple[np.ndarray, np.ndarray]:
    patch_grid_size = tuple(int(value) for value in motion_features["patch_grid_size"])
    patch_maps = patch_saliency_maps(motion_features)
    saliency_maps = resize_saliency_maps(patch_maps, patch_grid_size, image_shape_hw)
    masks = masks_from_saliency(saliency_maps, threshold_mode)
    return saliency_maps, masks
