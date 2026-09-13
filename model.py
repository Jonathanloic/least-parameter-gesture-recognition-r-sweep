from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """1D depthwise-separable block used by the Least Parameters baseline."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DepthwiseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, ...],
        kernel_size: int,
        p_drop: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for width in widths:
            layers.append(
                DepthwiseSeparableConv(
                    in_channels=prev_channels,
                    out_channels=width,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                )
            )
            prev_channels = width

        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(prev_channels, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def build_least_parameters_model(
    in_channels: int = 4,
    num_classes: int = 6,
) -> DepthwiseCNN:
    """Build the paper's Least Parameters configuration."""
    return DepthwiseCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        widths=(16, 16, 16, 16),
        kernel_size=9,
        p_drop=0.2,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Always derive the trainable parameter count from the instantiated model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
