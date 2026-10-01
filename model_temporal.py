from __future__ import annotations

import torch
from torch import nn

# The temporal-attention module is imported, not re-implemented, so that this
# model uses bit-for-bit the same operator as the temporal half of
# ChannelTemporalAttention1D in model_attention.py.
from model_attention import TemporalAttention1D


class TemporalDepthwiseSeparableConv(nn.Module):
    """Least Parameters depthwise-separable block with temporal attention after ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        p_drop: float = 0.2,
        temporal_kernel_size: int = 7,
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
        self.temporal = TemporalAttention1D(kernel_size=temporal_kernel_size)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        x = self.temporal(x)
        x = self.drop(x)
        return x


class TemporalDepthwiseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, ...],
        kernel_size: int,
        p_drop: float,
        temporal_kernel_size: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for width in widths:
            layers.append(
                TemporalDepthwiseSeparableConv(
                    in_channels=prev_channels,
                    out_channels=width,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                    temporal_kernel_size=temporal_kernel_size,
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


def build_least_parameters_temporal_model(
    in_channels: int = 4,
    num_classes: int = 6,
) -> TemporalDepthwiseCNN:
    """Least Parameters baseline + temporal attention only, after each block's ReLU."""
    return TemporalDepthwiseCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        widths=(16, 16, 16, 16),
        kernel_size=9,
        p_drop=0.2,
        temporal_kernel_size=7,
    )
