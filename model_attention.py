from __future__ import annotations

import torch
from torch import nn


class ChannelAttention1D(nn.Module):
    """Channel attention for 1D feature maps [B, C, T]."""

    def __init__(self, channels: int, reduction_ratio: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction_ratio)

        # Shared two-layer channel MLP implemented as 1x1 Conv1d.
        # bias=False matches the Best Accuracy attention implementation.
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = torch.mean(x, dim=2, keepdim=True)
        max_descriptor = torch.amax(x, dim=2, keepdim=True)

        weights = self.shared_mlp(avg_descriptor) + self.shared_mlp(max_descriptor)
        weights = self.sigmoid(weights)
        return x * weights


class TemporalAttention1D(nn.Module):
    """Temporal attention for 1D feature maps [B, C, T]."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Temporal-attention kernel size must be odd.")

        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = torch.mean(x, dim=1, keepdim=True)
        max_descriptor = torch.amax(x, dim=1, keepdim=True)
        descriptor = torch.cat([avg_descriptor, max_descriptor], dim=1)

        weights = self.sigmoid(self.conv(descriptor))
        return x * weights


class ChannelTemporalAttention1D(nn.Module):
    """Sequential channel attention followed by temporal attention."""

    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        temporal_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel = ChannelAttention1D(
            channels=channels,
            reduction_ratio=reduction_ratio,
        )
        self.temporal = TemporalAttention1D(kernel_size=temporal_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.temporal(x)
        return x


class AttentionDepthwiseSeparableConv(nn.Module):
    """Least Parameters depthwise-separable block with channel+temporal attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction_ratio: int,
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
        self.attention = ChannelTemporalAttention1D(
            channels=out_channels,
            reduction_ratio=reduction_ratio,
            temporal_kernel_size=temporal_kernel_size,
        )
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        x = self.attention(x)
        x = self.drop(x)
        return x


class AttentionDepthwiseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, ...],
        kernel_size: int,
        p_drop: float,
        reduction_ratio: int,
        temporal_kernel_size: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for width in widths:
            layers.append(
                AttentionDepthwiseSeparableConv(
                    in_channels=prev_channels,
                    out_channels=width,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                    reduction_ratio=reduction_ratio,
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


def build_least_parameters_channel_temporal_model(
    reduction_ratio: int,
    in_channels: int = 4,
    num_classes: int = 6,
) -> AttentionDepthwiseCNN:
    """Least Parameters baseline + channel and temporal attention after each ReLU."""
    return AttentionDepthwiseCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        widths=(16, 16, 16, 16),
        kernel_size=9,
        p_drop=0.2,
        reduction_ratio=reduction_ratio,
        temporal_kernel_size=7,
    )
