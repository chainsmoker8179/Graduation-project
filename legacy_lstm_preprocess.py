from __future__ import annotations

import torch
import torch.nn as nn


class RobustZScoreNormLayer(nn.Module):
    def __init__(self, center: torch.Tensor, scale: torch.Tensor, clip_outlier: bool) -> None:
        super().__init__()
        self.register_buffer("center", center.detach().clone().to(dtype=torch.float32))
        self.register_buffer("scale", scale.detach().clone().to(dtype=torch.float32))
        self.clip_outlier = bool(clip_outlier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = self.center.view(*([1] * (x.ndim - 1)), -1)
        safe_x = torch.where(torch.isnan(x), center, x)
        y = (safe_x - center) / self.scale
        if self.clip_outlier:
            y = torch.clamp(y, min=-3.0, max=3.0)
        return y


class FillnaLayer(nn.Module):
    def __init__(self, fill_value: float = 0.0) -> None:
        super().__init__()
        self.fill_value = float(fill_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fill = torch.full_like(x, self.fill_value)
        return torch.where(torch.isnan(x), fill, x)


__all__ = ["FillnaLayer", "RobustZScoreNormLayer"]
