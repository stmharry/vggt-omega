# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Sequence

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        self.output_device: torch.device | None = None

    def enable_head_parallelism(
        self,
        devices: Sequence[str | torch.device],
        *,
        include_camera_head: bool = True,
    ) -> None:
        self.aggregator.set_inter_frame_head_parallel_devices(devices)
        if include_camera_head and self.camera_head is not None:
            self.camera_head.set_head_parallel_devices(devices)

    def enable_global_inter_frame_query_parallelism(
        self,
        devices: Sequence[str | torch.device],
    ) -> None:
        self.aggregator.set_global_inter_frame_query_parallel_devices(devices)

    def enable_memory_parallelism(
        self,
        devices: Sequence[str | torch.device],
    ) -> None:
        parsed_devices = tuple(torch.device(device) for device in devices)
        if len(parsed_devices) < 2:
            raise ValueError("Memory-parallel inference requires at least two CUDA devices.")
        cache_device = parsed_devices[1]
        if cache_device.type != "cuda":
            raise ValueError(f"Memory-parallel inference requires a CUDA cache device, got {cache_device}.")

        self.aggregator.set_cache_device(cache_device)
        if self.camera_head is not None:
            self.camera_head.to(device=cache_device)
        if self.dense_head is not None:
            self.dense_head.to(device=cache_device)
        if self.text_alignment_head is not None:
            self.text_alignment_head.to(device=cache_device)

    def enable_balanced_memory_parallelism(
        self,
        devices: Sequence[str | torch.device],
        *,
        split_block: int = 23,
    ) -> None:
        parsed_devices = tuple(torch.device(device) for device in devices)
        if len(parsed_devices) < 2:
            raise ValueError("Balanced memory-parallel inference requires at least two CUDA devices.")
        primary_device, stage_device = parsed_devices[0], parsed_devices[1]
        if primary_device.type != "cuda" or stage_device.type != "cuda":
            raise ValueError(
                "Balanced memory-parallel inference requires CUDA devices, "
                f"got primary={primary_device}, stage={stage_device}."
            )

        self.aggregator.set_stage_devices(primary_device, stage_device, split_block)
        if self.camera_head is not None:
            self.camera_head.to(device=stage_device)
        if self.dense_head is not None:
            self.dense_head.to(device=stage_device)
        if self.text_alignment_head is not None:
            self.text_alignment_head.to(device=stage_device)

    def enable_pipeline_memory_parallelism(
        self,
        devices: Sequence[str | torch.device],
        *,
        stage_splits: Sequence[int],
        cache_device_index: int | None,
        cache_device: str | torch.device | None = None,
        head_device_index: int | None = None,
    ) -> None:
        parsed_devices = tuple(torch.device(device) for device in devices)
        if len(parsed_devices) < 2:
            raise ValueError("Pipeline memory-parallel inference requires at least two CUDA devices.")
        if any(device.type != "cuda" for device in parsed_devices):
            raise ValueError(f"Pipeline memory-parallel inference requires CUDA devices, got {parsed_devices}.")

        if cache_device is None:
            if cache_device_index is None:
                raise ValueError("cache_device_index is required when cache_device is not provided.")
            if cache_device_index < 0 or cache_device_index >= len(parsed_devices):
                raise ValueError(
                    f"cache_device_index must be in [0, {len(parsed_devices) - 1}], got {cache_device_index}."
                )
            parsed_cache_device = parsed_devices[cache_device_index]
            stage_devices = tuple(device for index, device in enumerate(parsed_devices) if index != cache_device_index)
        else:
            parsed_cache_device = torch.device(cache_device)
            stage_devices = parsed_devices

        if parsed_cache_device.type not in {"cuda", "cpu"}:
            raise ValueError(f"Pipeline cache device must be CUDA or CPU, got {parsed_cache_device}.")

        if head_device_index is None:
            head_device = parsed_cache_device if parsed_cache_device.type == "cuda" else parsed_devices[-1]
        else:
            if head_device_index < 0 or head_device_index >= len(parsed_devices):
                raise ValueError(
                    f"head_device_index must be in [0, {len(parsed_devices) - 1}], got {head_device_index}."
                )
            head_device = parsed_devices[head_device_index]

        self.aggregator.set_pipeline_stage_devices(
            stage_devices,
            stage_splits,
            cache_device=parsed_cache_device,
        )
        if self.camera_head is not None:
            self.camera_head.to(device=head_device)
        if self.dense_head is not None:
            self.dense_head.to(device=head_device)
        if self.text_alignment_head is not None:
            self.text_alignment_head.to(device=head_device)

    def set_patch_embed_chunk_size(self, chunk_size: int | None) -> None:
        self.aggregator.set_patch_embed_chunk_size(chunk_size)

    def set_output_device(self, device: str | torch.device | None) -> None:
        self.output_device = None if device is None else torch.device(device)
        if self.dense_head is not None:
            self.dense_head.set_output_device(device)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        camera_and_register_tokens = final_tokens[:, :, :patch_token_start].contiguous()
        if self.output_device is not None:
            camera_and_register_tokens = camera_and_register_tokens.to(
                device=self.output_device,
                non_blocking=self.output_device.type == "cuda",
            )
        predictions = {"camera_and_register_tokens": camera_and_register_tokens}
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )
                if self.output_device is not None:
                    pose_enc = pose_enc.to(device=self.output_device, non_blocking=self.output_device.type == "cuda")
                predictions["pose_enc"] = pose_enc

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = (
                images.to(device=self.output_device, non_blocking=self.output_device.type == "cuda")
                if self.output_device is not None
                else images
            )
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
