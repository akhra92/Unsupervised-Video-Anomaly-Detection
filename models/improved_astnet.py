import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wider_resnet import wresnet
from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights
from models.advanced_modules import TemporalAttention, MemoryModule, CBAM, SpatialAttention

logger = logging.getLogger(__name__)


class ImprovedASTNet(nn.Module):
    """
    Improved ASTNet with:
    - Temporal Attention (replacing simple TSM)
    - Memory Module for normal pattern storage
    - CBAM (Channel + Spatial Attention)
    - Enhanced skip connections
    """
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True, use_memory=True, use_temporal_attn=True):
        super(ImprovedASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME + '_improved'
        self.use_memory = use_memory
        self.use_temporal_attn = use_temporal_attn

        logger.info(f'=> {self.model_name}: Improved architecture with temporal attention and memory')

        # Encoder: WideResNet backbone
        self.wrn38 = wresnet(config, config.MODEL.NAME, pretrained=pretrained)

        channels = [4096, 2048, 1024, 512, 256, 128]

        # Temporal fusion with 1x1 convolutions
        self.conv_x8 = nn.Conv2d(channels[0] * frames, channels[1], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(channels[4] * frames, channels[4], kernel_size=1, bias=False)
        self.conv_x1 = nn.Conv2d(channels[5] * frames, channels[5], kernel_size=1, bias=False)

        # NEW: Temporal Attention Module
        if self.use_temporal_attn:
            self.temporal_attn = TemporalAttention(channels=channels[1], num_frames=frames)

        # NEW: Memory Module for normal patterns
        if self.use_memory:
            self.memory = MemoryModule(mem_size=2000, feature_dim=channels[1])

        # Decoder: Upsampling pathway
        self.up8 = ConvTransposeBnRelu(channels[1], channels[2], kernel_size=2)
        self.up4 = ConvTransposeBnRelu(channels[2] + channels[4], channels[3], kernel_size=2)
        self.up2 = ConvTransposeBnRelu(channels[3] + channels[5], channels[4], kernel_size=2)

        # NEW: CBAM Attention (Channel + Spatial) at each decoder stage
        self.cbam8 = CBAM(channels[2], reduction=16)
        self.cbam4 = CBAM(channels[3], reduction=16)
        self.cbam2 = CBAM(channels[4], reduction=16)

        # Final reconstruction layers
        self.final = nn.Sequential(
            ConvBnRelu(channels[4], channels[5], kernel_size=1, padding=0),
            ConvBnRelu(channels[5], channels[5], kernel_size=3, padding=1),
            nn.Conv2d(channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        # Initialize weights
        initialize_weights(self.conv_x1, self.conv_x2, self.conv_x8)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.cbam2, self.cbam4, self.cbam8)

    def forward(self, x, return_memory_attn=False):
        """
        Forward pass through improved architecture

        Args:
            x: List of input frames [frame1, frame2, ..., frameN]
            return_memory_attn: If True, return memory attention weights (for analysis)

        Returns:
            output: Predicted next frame
            (optional) memory_attn: Memory attention weights
        """
        # Extract features from each frame
        x1s, x2s, x8s = [], [], []
        for xi in x:
            x1, x2, x8 = self.wrn38(xi)
            x8s.append(x8)
            x2s.append(x2)
            x1s.append(x1)

        # Temporal fusion
        x8 = self.conv_x8(torch.cat(x8s, dim=1))
        x2 = self.conv_x2(torch.cat(x2s, dim=1))
        x1 = self.conv_x1(torch.cat(x1s, dim=1))

        # NEW: Apply temporal attention
        if self.use_temporal_attn:
            # Convert fused features back to list for temporal attention
            x8_attended = self.temporal_attn([x8s[i] for i in range(len(x8s))])
            x8 = x8 + x8_attended  # Residual connection

        # NEW: Memory module
        memory_attn = None
        if self.use_memory:
            retrieved, memory_attn = self.memory(x8, update_memory=self.training)
            x8 = x8 + 0.1 * retrieved  # Small contribution from memory

        # Decoder with CBAM attention
        x = self.up8(x8)
        x = self.cbam8(x)  # Channel + Spatial attention

        x = self.up4(torch.cat([x2, x], dim=1))
        x = self.cbam4(x)  # Channel + Spatial attention

        x = self.up2(torch.cat([x1, x], dim=1))
        x = self.cbam2(x)  # Channel + Spatial attention

        # Final reconstruction
        output = self.final(x)

        if return_memory_attn:
            return output, memory_attn
        return output


class ImprovedASTNetLarge(nn.Module):
    """
    Improved larger model for ShanghaiTech/Avenue datasets
    Based on wresnet2048_multiscale with improvements
    """
    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True, use_memory=True, use_temporal_attn=True):
        super(ImprovedASTNetLarge, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME + '_improved_large'
        self.use_memory = use_memory
        self.use_temporal_attn = use_temporal_attn

        logger.info(f'=> {self.model_name}: Large improved architecture')

        # Encoder
        self.wrn = wresnet(config, config.MODEL.NAME, pretrained=pretrained)

        channels = [4096, 2048, 1024, 512, 256, 128]

        # Temporal fusion
        self.conv_x7 = nn.Conv2d(channels[1] * frames, channels[0], kernel_size=1, bias=False)
        self.conv_x3 = nn.Conv2d(channels[4] * frames, channels[3], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(channels[5] * frames, channels[4], kernel_size=1, bias=False)

        # Temporal attention
        if self.use_temporal_attn:
            self.temporal_attn = TemporalAttention(channels=channels[0], num_frames=frames)

        # Memory module
        if self.use_memory:
            self.memory = MemoryModule(mem_size=3000, feature_dim=channels[0])

        # Decoder
        self.up8 = ConvTransposeBnRelu(channels[0], channels[1], kernel_size=2)
        self.up4 = ConvTransposeBnRelu(channels[2] + channels[3], channels[2], kernel_size=2)
        self.up2 = ConvTransposeBnRelu(channels[3] + channels[4], channels[3], kernel_size=2)

        # CBAM attention
        self.cbam8 = CBAM(channels[1], reduction=16)
        self.cbam4 = CBAM(channels[2], reduction=16)
        self.cbam2 = CBAM(channels[3], reduction=16)

        # Final layers
        self.final = nn.Sequential(
            ConvBnRelu(channels[3], channels[4], kernel_size=3, padding=1),
            ConvBnRelu(channels[4], channels[5], kernel_size=5, padding=2),
            nn.Conv2d(channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=(final_conv_kernel-1)//2,
                      bias=False)
        )

        # Initialize
        initialize_weights(self.conv_x2, self.conv_x3, self.conv_x7)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.cbam2, self.cbam4, self.cbam8)
        initialize_weights(self.final)

    def forward(self, x, return_memory_attn=False):
        # Extract features
        x2s, x3s, x7s = [], [], []
        for xi in x:
            x2, x3, x7 = self.wrn(xi)
            x7s.append(x7)
            x3s.append(x3)
            x2s.append(x2)

        # Temporal fusion
        x7 = self.conv_x7(torch.cat(x7s, dim=1))
        x3 = self.conv_x3(torch.cat(x3s, dim=1))
        x2 = self.conv_x2(torch.cat(x2s, dim=1))

        # Temporal attention
        if self.use_temporal_attn:
            x7_attended = self.temporal_attn([x7s[i] for i in range(len(x7s))])
            x7 = x7 + x7_attended

        # Memory
        memory_attn = None
        if self.use_memory:
            retrieved, memory_attn = self.memory(x7, update_memory=self.training)
            x7 = x7 + 0.1 * retrieved

        # Decoder
        x = self.up8(x7)
        x = self.cbam8(x)

        x = self.up4(torch.cat([x3, x], dim=1))
        x = self.cbam4(x)

        x = self.up2(torch.cat([x2, x], dim=1))
        x = self.cbam2(x)

        output = self.final(x)

        if return_memory_attn:
            return output, memory_attn
        return output
