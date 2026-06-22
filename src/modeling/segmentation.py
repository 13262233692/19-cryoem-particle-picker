import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from src.utils.logging import get_logger
from .backbone import LightResNet, DecoderBlock
from .attention import MaskedSelfAttention, CBAMBlock

logger = get_logger("modeling.segmentation")

class LightResUNet_MSA(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 base_channels: int = 32, num_res_blocks: int = 4,
                 attention_heads: int = 4, attention_channels: int = 64,
                 dropout_rate: float = 0.1,
                 use_deep_supervision: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.use_deep_supervision = use_deep_supervision
        self.encoder = LightResNet(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_res_blocks,
            dropout_rate=dropout_rate
        )
        encoder_channels = [base_channels] + self.encoder.channels
        self.attention_blocks = nn.ModuleList()
        self.attention_channels = min(encoder_channels[-1], attention_channels)
        for ch in reversed(encoder_channels[1:-1]):
            self.attention_blocks.append(
                MaskedSelfAttention(
                    channels=ch,
                    num_heads=attention_heads,
                    attention_channels=attention_channels,
                    drop_rate=dropout_rate
                )
            )
        self.cbam_blocks = nn.ModuleList()
        for ch in encoder_channels:
            self.cbam_blocks.append(CBAMBlock(channels=ch))
        decoder_channels = list(reversed(encoder_channels))
        self.decoders = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            skip_ch = decoder_channels[i + 1]
            out_ch = max(in_ch // 2, base_channels)
            self.decoders.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout_rate=dropout_rate
                )
            )
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=1)
        )
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for ch in decoder_channels[1:-1]:
                self.deep_supervision_heads.append(
                    nn.Sequential(
                        nn.Conv2d(ch, out_channels, kernel_size=1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    )
                )
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"LightResUNet_MSA initialized: "
                    f"in_channels={in_channels}, out_channels={out_channels}, "
                    f"base_channels={base_channels}, num_res_blocks={num_res_blocks}, "
                    f"attention_heads={attention_heads}, "
                    f"total_params={total_params/1e6:.2f}M, "
                    f"trainable_params={trainable_params/1e6:.2f}M")

    def _init_weights(self) -> None:
        for m in self.final_conv.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_deep_supervision:
            for head in self.deep_supervision_heads:
                for m in head.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        enhanced_features = []
        for feat, cbam in zip(features, self.cbam_blocks):
            enhanced_features.append(cbam(feat))
        bottleneck = enhanced_features[-1]
        for i, attn in enumerate(self.attention_blocks):
            feat_idx = len(enhanced_features) - 2 - i
            if feat_idx >= 1:
                enhanced_features[feat_idx] = attn(enhanced_features[feat_idx])
        deep_outputs = []
        x = bottleneck
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(enhanced_features) - 2 - i
            if skip_idx >= 0:
                skip = enhanced_features[skip_idx]
            else:
                skip = enhanced_features[0]
            x = decoder(x, skip)
            if self.use_deep_supervision and i < len(self.deep_supervision_heads):
                deep_out = self.deep_supervision_heads[i](x)
                deep_outputs.append(deep_out)
        output = self.final_conv(x)
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_deep_supervision and self.training:
            deep_outputs.append(output)
            return tuple(deep_outputs)
        return output

    def predict(self, x: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if isinstance(logits, list):
                logits = logits[-1]
            if self.out_channels > 1:
                probs = F.softmax(logits, dim=1)
                mask = probs[:, 1, :, :] > threshold
            else:
                probs = torch.sigmoid(logits)
                mask = probs > threshold
        return mask.float()

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.eval()
        with torch.no_grad():
            features = self.encoder(x)
            enhanced_features = []
            for feat, cbam in zip(features, self.cbam_blocks):
                enhanced_features.append(cbam(feat))
            attention_maps = []
            for i, attn in enumerate(self.attention_blocks):
                feat_idx = len(enhanced_features) - 2 - i
                if feat_idx >= 1:
                    feat = enhanced_features[feat_idx]
                    mask = attn.create_content_mask(feat)
                    attention_maps.append(mask.squeeze(1).mean(dim=1))
                    enhanced_features[feat_idx] = attn(feat, mask)
        return attention_maps

    def get_config(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "num_res_blocks": self.num_res_blocks,
            "use_deep_supervision": self.use_deep_supervision,
        }

    def load_checkpoint(self, checkpoint_path: str,
                       map_location: Optional[str] = None) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")

    def save_checkpoint(self, save_path: str,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: int = 0,
                       metrics: Optional[Dict[str, float]] = None) -> str:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
            'epoch': epoch,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to: {save_path}")
        return save_path
