# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Sequence, Tuple

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from .utils import cat_keep_shapes, uncat_with_shapes


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        # VGGT-Omega change: the aggregator checkpoint was trained with Q/K
        # normalization, while upstream DINOv3 attention does not expose it.
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)
        self.projected_head_parallel_devices: tuple[torch.device, ...] = ()

    def set_projected_head_parallel_devices(self, devices: Sequence[str | torch.device] | None) -> None:
        if devices is None:
            self.projected_head_parallel_devices = ()
            return
        parsed_devices = tuple(torch.device(device) for device in devices)
        if len(parsed_devices) <= 1:
            self.projected_head_parallel_devices = ()
            return
        if self.num_heads % len(parsed_devices) != 0:
            raise ValueError(
                "Projected head-parallel attention requires the number of attention heads "
                f"({self.num_heads}) to be divisible by the device count ({len(parsed_devices)})."
            )
        for device in parsed_devices:
            if device.type != "cuda":
                raise ValueError(f"Projected head-parallel attention requires CUDA devices, got {device}.")
        self.projected_head_parallel_devices = parsed_devices

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        if self.projected_head_parallel_devices:
            if attn_bias is not None:
                raise ValueError("Projected head-parallel attention does not support attn_bias.")
            if rope is not None:
                raise ValueError("Projected head-parallel attention is only enabled for non-RoPE all-frame blocks.")
            return self.forward_projected_head_parallel(x)
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        if self.projected_head_parallel_devices:
            if attn_bias is not None:
                raise ValueError("Projected head-parallel attention does not support attn_bias.")
            if any(rope is not None for rope in rope_list):
                raise ValueError("Projected head-parallel attention is only enabled for non-RoPE all-frame blocks.")
            return [self.forward_projected_head_parallel(x) for x in x_list]
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])

    def forward_projected_head_parallel(self, x: Tensor) -> Tensor:
        source_device = x.device
        B, N, C = x.shape
        head_dim = C // self.num_heads
        heads_per_device = self.num_heads // len(self.projected_head_parallel_devices)
        output = None
        qkv_bias = self._effective_qkv_bias()

        for shard_idx, device in enumerate(self.projected_head_parallel_devices):
            head_start = shard_idx * heads_per_device
            head_end = head_start + heads_per_device
            column_start = head_start * head_dim
            column_end = head_end * head_dim

            q_rows = torch.arange(column_start, column_end, device=self.qkv.weight.device)
            k_rows = torch.arange(C + column_start, C + column_end, device=self.qkv.weight.device)
            v_rows = torch.arange(2 * C + column_start, 2 * C + column_end, device=self.qkv.weight.device)
            rows = torch.cat([q_rows, k_rows, v_rows])

            x_device = x.to(device=device, non_blocking=True)
            qkv_weight = self.qkv.weight.index_select(0, rows).to(device=device, non_blocking=True)
            if qkv_bias is None:
                shard_bias = None
            else:
                shard_bias = qkv_bias.index_select(0, rows).to(device=device, non_blocking=True)

            qkv = F.linear(x_device, qkv_weight, shard_bias)
            qkv = qkv.reshape(B, N, 3, heads_per_device, head_dim)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [tensor.transpose(1, 2) for tensor in [q, k, v]]

            if self.use_qk_norm:
                q = F.layer_norm(
                    q,
                    (head_dim,),
                    self.q_norm.weight.to(device=device, dtype=q.dtype, non_blocking=True),
                    self.q_norm.bias.to(device=device, dtype=q.dtype, non_blocking=True),
                    self.q_norm.eps,
                )
                k = F.layer_norm(
                    k,
                    (head_dim,),
                    self.k_norm.weight.to(device=device, dtype=k.dtype, non_blocking=True),
                    self.k_norm.bias.to(device=device, dtype=k.dtype, non_blocking=True),
                    self.k_norm.eps,
                )

            attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn_v = attn_v.transpose(1, 2).reshape(B, N, column_end - column_start)
            proj_weight = self.proj.weight[:, column_start:column_end].to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=False):
                partial = F.linear(attn_v.float(), proj_weight.float(), None)
            partial = partial.to(device=source_device, non_blocking=True)
            output = partial if output is None else output + partial

        if output is None:
            raise RuntimeError("Projected head-parallel attention has no configured devices.")
        if self.proj.bias is not None:
            output = output + self.proj.bias.to(device=source_device, dtype=output.dtype)
        return self.proj_drop(output.to(dtype=x.dtype))

    def _effective_qkv_bias(self) -> Tensor | None:
        if self.qkv.bias is None:
            return None
        bias_mask = getattr(self.qkv, "bias_mask", None)
        if bias_mask is None:
            return self.qkv.bias
        return self.qkv.bias * bias_mask.to(dtype=self.qkv.bias.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
