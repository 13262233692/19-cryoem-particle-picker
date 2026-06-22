import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from src.utils.logging import get_logger

logger = get_logger("modeling.attention")

class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_channels = channels // num_heads
        self.scale = self.head_channels ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self._init_weights()
        logger.info(f"MultiHeadAttention initialized: channels={channels}, "
                    f"num_heads={num_heads}, head_channels={self.head_channels}")

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_channels, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, self.num_heads * self.head_channels, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4,
                 attention_channels: int = 64,
                 qkv_bias: bool = True, drop_rate: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attention_channels = attention_channels
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MultiHeadAttention(
            channels=min(channels, attention_channels),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            proj_drop=drop_rate
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.Dropout(drop_rate)
        )
        self.channel_proj = nn.Conv2d(channels, min(channels, attention_channels), kernel_size=1)
        self.channel_back = nn.Conv2d(min(channels, attention_channels), channels, kernel_size=1)
        self._init_weights()
        logger.info(f"MaskedSelfAttention initialized: channels={channels}, "
                    f"num_heads={num_heads}, attention_channels={attention_channels}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def create_content_mask(self, x: torch.Tensor,
                           threshold: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            content = x.mean(dim=1, keepdim=True)
            threshold_val = content.mean() + threshold * content.std()
            mask = (content > threshold_val).float()
            mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='nearest')
            mask = mask.squeeze(1)
            B, H, W = mask.shape
            mask_2d = mask.reshape(B, H * W, 1) * mask.reshape(B, 1, H * W)
            mask_2d = mask_2d.unsqueeze(1)
            return mask_2d

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if mask is None:
            mask = self.create_content_mask(x)
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_norm).permute(0, 3, 1, 2).contiguous()
        x_proj = self.channel_proj(x_norm)
        attn_out = self.attn(x_proj, mask)
        attn_out = self.channel_back(attn_out)
        x = x + attn_out
        x_norm2 = x.permute(0, 2, 3, 1).contiguous()
        x_norm2 = self.norm2(x_norm2).permute(0, 3, 1, 2).contiguous()
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.conv(combined))
        return x * scale

class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        logger.info(f"CBAMBlock initialized: channels={channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
