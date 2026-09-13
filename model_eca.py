from __future__ import annotations

import math

import torch
from torch import nn


def adaptive_eca_kernel_size(
    channels: int,
    gamma: float = 2.0,
    b: float = 1.0,
) -> int:
    """Return the ECA-Net adaptive odd channel kernel size.

    ECA-Net uses an adaptive mapping from channel count C to an odd 1D
    convolution kernel. We keep the same gamma=2 and b=1 configuration used
    in the Best Accuracy + ECA implementation.
    """
    if channels <= 0:
        raise ValueError("channels must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    value = (math.log2(channels) + b) / gamma

    # Nearest positive odd integer; ties are resolved upward.
    lower = max(1, int(math.floor(value)))
    upper = max(1, int(math.ceil(value)))

    odd_candidates: set[int] = set()
    for candidate in (lower - 1, lower, lower + 1, upper - 1, upper, upper + 1):
        if candidate > 0 and candidate % 2 == 1:
            odd_candidates.add(candidate)

    return min(odd_candidates, key=lambda k: (abs(k - value), -k))


class ECAChannelAttention1D(nn.Module):
    """ECA channel attention adapted to 1D feature maps [B, C, T].

    Global average pooling compresses the temporal dimension T. The ECA
    Conv1d then operates across the channel descriptor C, not across time.
    """

    def __init__(
        self,
        channels: int,
        gamma: float = 2.0,
        b: float = 1.0,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.gamma = gamma
        self.b = b
        self.kernel_size = adaptive_eca_kernel_size(channels, gamma=gamma, b=b)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        y = self.avg_pool(x)   # [B, C, 1]
        y = y.transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)       # local cross-channel interaction
        y = self.sigmoid(y)
        y = y.transpose(1, 2)  # [B, C, 1]
        return x * y


class ECADepthwiseSeparableConv(nn.Module):
    """Least Parameters block with ECA at the same attention position.

    Depthwise -> Pointwise -> BatchNorm -> MaxPool -> ReLU -> ECA -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        p_drop: float = 0.2,
        eca_gamma: float = 2.0,
        eca_b: float = 1.0,
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
        self.eca = ECAChannelAttention1D(
            channels=out_channels,
            gamma=eca_gamma,
            b=eca_b,
        )
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        x = self.eca(x)
        x = self.drop(x)
        return x


class ECADepthwiseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, ...],
        kernel_size: int,
        p_drop: float,
        eca_gamma: float = 2.0,
        eca_b: float = 1.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for width in widths:
            layers.append(
                ECADepthwiseSeparableConv(
                    in_channels=prev_channels,
                    out_channels=width,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                    eca_gamma=eca_gamma,
                    eca_b=eca_b,
                )
            )
            prev_channels = width

        self.features = nn.Sequential(*layers)

        # Classifier is unchanged from the Least Parameters baseline.
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


def build_least_parameters_eca_model(
    in_channels: int = 4,
    num_classes: int = 6,
) -> ECADepthwiseCNN:
    """Least Parameters baseline + ECA after every block ReLU."""
    return ECADepthwiseCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        widths=(16, 16, 16, 16),
        kernel_size=9,
        p_drop=0.2,
        eca_gamma=2.0,
        eca_b=1.0,
    )
