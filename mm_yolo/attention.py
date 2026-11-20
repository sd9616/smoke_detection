"""Cross-attention fusion between RGB and thermal feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Bidirectional multi-head cross-attention, mixed back to one feature map.

    Queries from one modality attend to keys and values from the other. Tokens
    are pooled to a small grid so the P3 map (80×80 at 640 input) stays cheap.

    The 1×1 mix starts as "pass RGB through, add a tiny thermal residual", so a
    model loaded from the existing smoke checkpoint still behaves like that
    detector until training moves the fusion weights.
    """

    def __init__(self, channels: int, heads: int = 4, max_tokens: int = 16):
        super().__init__()
        if channels < heads:
            heads = 1
        dim = max(heads, (channels // 2) // heads * heads)
        self.heads = heads
        self.max_tokens = max_tokens
        self.dim = dim
        self.head_dim = dim // heads

        self.q_rgb = nn.Conv2d(channels, dim, 1, bias=False)
        self.kv_thermal = nn.Conv2d(channels, dim * 2, 1, bias=False)
        self.q_thermal = nn.Conv2d(channels, dim, 1, bias=False)
        self.kv_rgb = nn.Conv2d(channels, dim * 2, 1, bias=False)
        self.out_rgb = nn.Conv2d(dim, channels, 1, bias=False)
        self.out_thermal = nn.Conv2d(dim, channels, 1, bias=False)
        self.mix = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self._init_reuse_rgb(channels)

    def _init_reuse_rgb(self, channels: int) -> None:
        nn.init.zeros_(self.out_rgb.weight)
        nn.init.zeros_(self.out_thermal.weight)
        nn.init.zeros_(self.mix.weight)
        with torch.no_grad():
            for i in range(channels):
                self.mix.weight[i, i, 0, 0] = 1.0
                self.mix.weight[i, channels + i, 0, 0] = 1e-3

    def _attend(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        q_conv: nn.Conv2d,
        kv_conv: nn.Conv2d,
    ) -> torch.Tensor:
        batch, _, height, width = query.shape
        grid_h = min(self.max_tokens, height)
        grid_w = min(self.max_tokens, width)
        q = F.adaptive_avg_pool2d(q_conv(query), (grid_h, grid_w))
        k, v = F.adaptive_avg_pool2d(kv_conv(context), (grid_h, grid_w)).chunk(2, dim=1)

        def tokens(t: torch.Tensor) -> torch.Tensor:
            t = t.view(batch, self.heads, self.head_dim, grid_h * grid_w)
            return t.permute(0, 1, 3, 2)

        q_t, k_t, v_t = tokens(q), tokens(k), tokens(v)
        attn = torch.softmax(q_t @ k_t.transpose(-2, -1) * (self.head_dim**-0.5), dim=-1)
        out = (attn @ v_t).permute(0, 1, 3, 2).contiguous().view(batch, self.dim, grid_h, grid_w)
        return F.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        rgb_h = rgb + self.out_rgb(self._attend(rgb, thermal, self.q_rgb, self.kv_thermal))
        thermal_h = thermal + self.out_thermal(self._attend(thermal, rgb, self.q_thermal, self.kv_rgb))
        return self.mix(torch.cat((rgb_h, thermal_h), dim=1))
