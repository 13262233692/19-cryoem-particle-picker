import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from src.utils.logging import get_logger

logger = get_logger("modeling.backbone")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)

class LightResNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 num_blocks: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.channels = [base_channels * (2 ** i) for i in range(num_blocks)]
        self.layers = nn.ModuleList()
        current_channels = base_channels
        for i in range(num_blocks):
            stride = 2 if i > 0 else 1
            self.layers.append(ResidualBlock(
                current_channels, self.channels[i],
                stride=stride, dropout_rate=dropout_rate
            ))
            current_channels = self.channels[i]
        self._init_weights()
        logger.info(f"LightResNet initialized: in_channels={in_channels}, "
                    f"base_channels={base_channels}, num_blocks={num_blocks}, "
                    f"output_channels={self.channels}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.stem(x)
        features.append(x)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
