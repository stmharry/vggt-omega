# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
import torch.nn as nn

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.cache_device: torch.device | None = None
        self.patch_embed_chunk_size: int | None = None
        self.stage_devices: tuple[torch.device, ...] = ()
        self.stage_split_blocks: tuple[int, ...] = ()
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def set_inter_frame_head_parallel_devices(
        self,
        devices: Sequence[str | torch.device] | None,
    ) -> None:
        for block in self.inter_frame_blocks:
            block.set_head_parallel_devices(devices)

    def set_global_inter_frame_query_blockwise_devices(
        self,
        devices: Sequence[str | torch.device] | None,
        *,
        query_block_size: int = 2048,
    ) -> None:
        for block, attention_type in zip(self.inter_frame_blocks, self.inter_frame_attention_types):
            block.set_query_blockwise_devices(
                devices if attention_type == "global" else None,
                query_block_size=query_block_size,
            )

    def set_global_inter_frame_context_parallel_devices(
        self,
        devices: Sequence[str | torch.device] | None,
        *,
        query_block_size: int = 8192,
        key_block_size: int = 4096,
        implementation: str = "gather-sdpa",
    ) -> None:
        for block, attention_type in zip(self.inter_frame_blocks, self.inter_frame_attention_types):
            block.set_context_parallel_devices(
                devices if attention_type == "global" else None,
                query_block_size=query_block_size,
                key_block_size=key_block_size,
                implementation=implementation,
            )

    def set_cache_device(self, device: str | torch.device | None) -> None:
        self.cache_device = None if device is None else torch.device(device)

    def set_patch_embed_chunk_size(self, chunk_size: int | None) -> None:
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"patch_embed_chunk_size must be positive, got {chunk_size}.")
        self.patch_embed_chunk_size = chunk_size

    def set_stage_devices(
        self,
        primary_device: str | torch.device,
        stage_device: str | torch.device,
        split_block: int,
    ) -> None:
        self.set_pipeline_stage_devices(
            [primary_device, stage_device],
            [split_block],
            cache_device=stage_device,
        )

    def set_pipeline_stage_devices(
        self,
        stage_devices: Sequence[str | torch.device],
        split_blocks: Sequence[int],
        *,
        cache_device: str | torch.device,
    ) -> None:
        parsed_stage_devices = tuple(torch.device(device) for device in stage_devices)
        parsed_cache_device = torch.device(cache_device)
        if len(parsed_stage_devices) < 2:
            raise ValueError("Pipeline memory-parallel inference requires at least two stage devices.")
        if len(split_blocks) != len(parsed_stage_devices) - 1:
            raise ValueError(
                "Pipeline memory-parallel inference requires exactly one fewer split "
                f"than stage devices, got devices={len(parsed_stage_devices)} splits={len(split_blocks)}."
            )
        if any(device.type != "cuda" for device in parsed_stage_devices):
            raise ValueError(
                "Pipeline memory-parallel inference requires CUDA stage devices, "
                f"got stage_devices={parsed_stage_devices}, cache_device={parsed_cache_device}."
            )
        if parsed_cache_device.type not in {"cuda", "cpu"}:
            raise ValueError(f"Pipeline cache device must be CUDA or CPU, got {parsed_cache_device}.")
        parsed_split_blocks = tuple(int(split_block) for split_block in split_blocks)
        if tuple(sorted(parsed_split_blocks)) != parsed_split_blocks or len(set(parsed_split_blocks)) != len(
            parsed_split_blocks
        ):
            raise ValueError(f"split_blocks must be sorted and unique, got {parsed_split_blocks}.")
        for split_block in parsed_split_blocks:
            if split_block < 0 or split_block > self.depth:
                raise ValueError(f"split_blocks must be in [0, {self.depth}], got {parsed_split_blocks}.")

        self.stage_devices = parsed_stage_devices
        self.stage_split_blocks = parsed_split_blocks
        self.cache_device = parsed_cache_device
        for block_idx in range(self.depth):
            block_device = self._stage_device_for_block(block_idx)
            self.frame_blocks[block_idx].to(device=block_device)
            self.inter_frame_blocks[block_idx].to(device=block_device)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        patch_tokens = self._run_patch_embed(images, batch_size, num_frames)
        first_block_device = self._stage_device_for_block(0)
        if patch_tokens.device != first_block_device:
            patch_tokens = patch_tokens.to(device=first_block_device, non_blocking=True)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)
        if camera_token.device != first_block_device:
            camera_token = camera_token.to(device=first_block_device, non_blocking=True)
            register_token = register_token.to(device=first_block_device, non_blocking=True)

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=first_block_device, dtype=torch.float32),
                rope_cos.to(device=first_block_device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.depth):
            if block_idx in self.stage_split_blocks:
                stage_device = self._stage_device_for_block(block_idx)
                tokens = tokens.to(device=stage_device, non_blocking=True)
                frame_rope = (
                    frame_rope[0].to(device=stage_device, non_blocking=True),
                    frame_rope[1].to(device=stage_device, non_blocking=True),
                )
            should_cache_layer = block_idx in self.cached_layer_indices
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
                return_frame_tokens=should_cache_layer,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )
            if should_cache_layer:
                if frame_tokens is None:
                    raise RuntimeError(f"Expected cached frame tokens for block {block_idx}")
                cached_tokens = torch.cat([frame_tokens, tokens], dim=-1)
                if self.cache_device is not None:
                    cached_tokens = cached_tokens.to(
                        device=self.cache_device,
                        non_blocking=self.cache_device.type == "cuda",
                    )
                outputs.append(cached_tokens)
            else:
                outputs.append(None)

        return outputs, self.patch_token_start

    def _run_patch_embed(self, images: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        _, _, num_channels, height, width = images.shape
        patch_device = next(self.patch_embed.parameters()).device
        flat_images = images.view(batch_size * num_frames, num_channels, height, width)
        mean = self._resnet_mean.view(1, 3, 1, 1).to(device=patch_device)
        std = self._resnet_std.view(1, 3, 1, 1).to(device=patch_device)

        if self.patch_embed_chunk_size is None or self.patch_embed_chunk_size >= flat_images.shape[0]:
            normalized_images = flat_images.to(device=patch_device, non_blocking=True)
            normalized_images = (normalized_images - mean) / std
            return self._call_patch_embed(normalized_images)

        patch_token_chunks = []
        for image_chunk in flat_images.split(self.patch_embed_chunk_size, dim=0):
            image_chunk = image_chunk.to(device=patch_device, non_blocking=True)
            image_chunk = (image_chunk - mean) / std
            patch_token_chunks.append(self._call_patch_embed(image_chunk))
        return torch.cat(patch_token_chunks, dim=0)

    def _call_patch_embed(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            return patch_tokens["x_norm_patchtokens"]
        return patch_tokens

    def _stage_device_for_block(self, block_idx: int) -> torch.device:
        if not self.stage_devices:
            return next(self.frame_blocks[block_idx].parameters()).device
        stage_index = sum(split_block <= block_idx for split_block in self.stage_split_blocks)
        return self.stage_devices[stage_index]

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
        *,
        return_frame_tokens: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        if not return_frame_tokens:
            return tokens, None
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
