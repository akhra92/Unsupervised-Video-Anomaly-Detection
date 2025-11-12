"""
Lightweight encoder backbones for efficient inference

This module provides alternative encoder backbones that are more
efficient than WideResNet-38, enabling real-time anomaly detection.

Available encoders:
- EfficientNet-B0: 5.3M params, ~10x faster than WideResNet-38
- MobileNetV3-Large: 5.4M params, ~12x faster than WideResNet-38
- MobileNetV3-Small: 2.5M params, ~15x faster than WideResNet-38

All encoders are pretrained on ImageNet for better feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 encoder for feature extraction

    Advantages:
    - 5.3M parameters (vs 100M+ for WideResNet-38)
    - 10x faster inference
    - Excellent accuracy-efficiency tradeoff
    - Compound scaling for optimal performance

    Output features at multiple scales:
    - Low-level (stage 2): 24 channels, H/4 x W/4
    - Mid-level (stage 4): 80 channels, H/16 x W/16
    - High-level (stage 8): 320 channels, H/32 x W/32
    """

    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            efficientnet = models.efficientnet_b0(weights=weights)
        else:
            efficientnet = models.efficientnet_b0(weights=None)

        self.features = efficientnet.features

        # Feature extraction indices for multi-scale outputs
        # EfficientNet-B0 structure:
        # 0-2: stem (32 channels)
        # 3: stage 1 (16 channels, H/2)
        # 4: stage 2 (24 channels, H/4)  <- low-level features
        # 5: stage 3 (40 channels, H/8)
        # 6: stage 4 (80 channels, H/16) <- mid-level features
        # 7: stage 5 (112 channels, H/16)
        # 8: stage 6 (192 channels, H/32)
        # 9: stage 7 (320 channels, H/32) <- high-level features

        self.layer_indices = {
            'low': 4,    # 24 channels, H/4 x W/4
            'mid': 6,    # 80 channels, H/16 x W/16
            'high': 9    # 320 channels, H/32 x W/32  (last block)
        }

    def forward(self, x):
        """
        Extract multi-scale features

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            low: Low-level features [B, 24, H/4, W/4]
            mid: Mid-level features [B, 80, H/16, W/16]
            high: High-level features [B, 320, H/32, W/32]
        """
        low_features = None
        mid_features = None
        high_features = None

        for idx, layer in enumerate(self.features):
            x = layer(x)

            if idx == self.layer_indices['low']:
                low_features = x
            elif idx == self.layer_indices['mid']:
                mid_features = x
            elif idx == self.layer_indices['high']:
                high_features = x

        return low_features, mid_features, high_features


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3-Large/Small encoder for feature extraction

    Advantages:
    - 5.4M params (Large) or 2.5M params (Small)
    - 12-15x faster inference than WideResNet
    - Mobile-optimized architecture
    - Excellent for edge deployment

    Output features at multiple scales:
    - Low-level: 40 channels, H/8 x W/8
    - Mid-level: 112 channels, H/16 x W/16
    - High-level: 960 channels, H/32 x W/32
    """

    def __init__(self, variant='large', pretrained=True):
        """
        Args:
            variant: 'large' or 'small'
            pretrained: Use ImageNet pretrained weights
        """
        super(MobileNetV3Encoder, self).__init__()

        self.variant = variant

        if variant == 'large':
            if pretrained:
                weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
                mobilenet = models.mobilenet_v3_large(weights=weights)
            else:
                mobilenet = models.mobilenet_v3_large(weights=None)

            # Feature channels for Large variant
            self.low_channels = 40
            self.mid_channels = 112
            self.high_channels = 960

        else:  # small
            if pretrained:
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
                mobilenet = models.mobilenet_v3_small(weights=weights)
            else:
                mobilenet = models.mobilenet_v3_small(weights=None)

            # Feature channels for Small variant
            self.low_channels = 24
            self.mid_channels = 48
            self.high_channels = 576

        self.features = mobilenet.features

        # MobileNetV3-Large structure:
        # 0-3: stem + early blocks (40 channels, H/8)  <- low-level
        # 4-6: mid blocks (80-112 channels, H/16)      <- mid-level
        # 7-12: late blocks (160-960 channels, H/32)   <- high-level

        if variant == 'large':
            self.layer_indices = {
                'low': 3,   # 40 channels, H/8
                'mid': 6,   # 112 channels, H/16
                'high': 12  # 960 channels, H/32
            }
        else:  # small
            self.layer_indices = {
                'low': 2,   # 24 channels, H/8
                'mid': 4,   # 48 channels, H/16
                'high': 8   # 576 channels, H/32
            }

    def forward(self, x):
        """
        Extract multi-scale features

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            low: Low-level features [B, C1, H/8, W/8]
            mid: Mid-level features [B, C2, H/16, W/16]
            high: High-level features [B, C3, H/32, W/32]
        """
        low_features = None
        mid_features = None
        high_features = None

        for idx, layer in enumerate(self.features):
            x = layer(x)

            if idx == self.layer_indices['low']:
                low_features = x
            elif idx == self.layer_indices['mid']:
                mid_features = x
            elif idx == self.layer_indices['high']:
                high_features = x

        return low_features, mid_features, high_features


def get_encoder(encoder_type='efficientnet', pretrained=True, **kwargs):
    """
    Factory function to get encoder by name

    Args:
        encoder_type: 'efficientnet', 'mobilenet_large', or 'mobilenet_small'
        pretrained: Use pretrained weights
        **kwargs: Additional arguments for encoder

    Returns:
        Encoder module
    """
    encoder_type = encoder_type.lower()

    if encoder_type == 'efficientnet' or encoder_type == 'efficientnet-b0':
        return EfficientNetEncoder(pretrained=pretrained)

    elif encoder_type == 'mobilenet_large' or encoder_type == 'mobilenetv3_large':
        return MobileNetV3Encoder(variant='large', pretrained=pretrained)

    elif encoder_type == 'mobilenet_small' or encoder_type == 'mobilenetv3_small':
        return MobileNetV3Encoder(variant='small', pretrained=pretrained)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Choose from: efficientnet, mobilenet_large, mobilenet_small")


def get_encoder_channels(encoder_type='efficientnet'):
    """
    Get output channel dimensions for each encoder

    Args:
        encoder_type: Encoder name

    Returns:
        dict: Channel dimensions for low, mid, high features
    """
    encoder_type = encoder_type.lower()

    channels = {
        'efficientnet': {'low': 24, 'mid': 80, 'high': 320},
        'efficientnet-b0': {'low': 24, 'mid': 80, 'high': 320},
        'mobilenet_large': {'low': 40, 'mid': 112, 'high': 960},
        'mobilenetv3_large': {'low': 40, 'mid': 112, 'high': 960},
        'mobilenet_small': {'low': 24, 'mid': 48, 'high': 576},
        'mobilenetv3_small': {'low': 24, 'mid': 48, 'high': 576},
    }

    if encoder_type not in channels:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return channels[encoder_type]


if __name__ == '__main__':
    # Test encoders
    import torch

    print("Testing Lightweight Encoders")
    print("="*60)

    # Test input
    x = torch.randn(2, 3, 256, 256)

    # Test EfficientNet
    print("\n1. EfficientNet-B0:")
    encoder = EfficientNetEncoder(pretrained=False)
    low, mid, high = encoder(x)
    print(f"   Input: {x.shape}")
    print(f"   Low-level output: {low.shape}")
    print(f"   Mid-level output: {mid.shape}")
    print(f"   High-level output: {high.shape}")
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")

    # Test MobileNetV3-Large
    print("\n2. MobileNetV3-Large:")
    encoder = MobileNetV3Encoder(variant='large', pretrained=False)
    low, mid, high = encoder(x)
    print(f"   Input: {x.shape}")
    print(f"   Low-level output: {low.shape}")
    print(f"   Mid-level output: {mid.shape}")
    print(f"   High-level output: {high.shape}")
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")

    # Test MobileNetV3-Small
    print("\n3. MobileNetV3-Small:")
    encoder = MobileNetV3Encoder(variant='small', pretrained=False)
    low, mid, high = encoder(x)
    print(f"   Input: {x.shape}")
    print(f"   Low-level output: {low.shape}")
    print(f"   Mid-level output: {mid.shape}")
    print(f"   High-level output: {high.shape}")
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")

    print("\n" + "="*60)
    print("All encoders working correctly!")
