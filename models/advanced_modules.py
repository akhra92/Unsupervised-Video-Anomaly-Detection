import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism that learns to focus on relevant past frames
    for predicting the next frame. More sophisticated than TSM.
    """
    def __init__(self, channels, num_frames=4, reduction=8):
        super(TemporalAttention, self).__init__()
        self.num_frames = num_frames
        self.channels = channels

        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # Temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1) * (channels // reduction) ** 0.5)

    def forward(self, features_list):
        """
        Args:
            features_list: List of T feature maps, each [B, C, H, W]
        Returns:
            Temporally attended features [B, C, H, W]
        """
        B, C, H, W = features_list[0].shape
        T = len(features_list)

        # Stack features: [B, T, C, H, W]
        features = torch.stack(features_list, dim=1)

        # Use last frame as query
        query_feat = features[:, -1]  # [B, C, H, W]
        Q = self.query(query_feat)  # [B, C//r, H, W]
        Q = Q.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//r]

        # All frames as keys and values
        K_list = []
        V_list = []
        for i in range(T):
            K = self.key(features[:, i])  # [B, C//r, H, W]
            V = self.value(features[:, i])  # [B, C, H, W]
            K_list.append(K.view(B, -1, H * W).permute(0, 2, 1))  # [B, HW, C//r]
            V_list.append(V.view(B, -1, H * W).permute(0, 2, 1))  # [B, HW, C]

        K = torch.stack(K_list, dim=1)  # [B, T, HW, C//r]
        V = torch.stack(V_list, dim=1)  # [B, T, HW, C]

        # Compute attention: Q @ K^T
        # [B, HW, C//r] @ [B, T, C//r, HW] -> [B, HW, T, HW]
        attention_scores = torch.einsum('bnc,btmc->bntm', Q, K.permute(0, 1, 3, 2))
        attention_scores = attention_scores / self.temperature

        # Softmax over temporal and spatial dimensions
        attention_weights = F.softmax(attention_scores.view(B, H * W, -1), dim=-1)
        attention_weights = attention_weights.view(B, H * W, T, H * W)

        # Apply attention to values
        # [B, HW, T, HW] @ [B, T, HW, C] -> [B, HW, C]
        attended = torch.einsum('bntm,btmc->bnc', attention_weights, V)

        # Reshape and project
        attended = attended.permute(0, 2, 1).view(B, C, H, W)
        output = self.out_proj(attended)

        # Residual connection
        return output + features[:, -1]


class MemoryModule(nn.Module):
    """
    Memory module that stores prototypical normal patterns.
    Anomalies will have poor matches with memory, leading to higher errors.
    """
    def __init__(self, mem_size=2000, feature_dim=2048, shrink_thres=0.0025):
        super(MemoryModule, self).__init__()
        self.mem_size = mem_size
        self.feature_dim = feature_dim
        self.shrink_thres = shrink_thres

        # Initialize memory bank
        self.register_buffer('memory', torch.randn(mem_size, feature_dim))
        self.memory = F.normalize(self.memory, dim=1)

        # Learnable query and memory update
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.alpha = 0.1  # Memory update rate

    def forward(self, features, update_memory=True):
        """
        Args:
            features: [B, C, H, W]
            update_memory: Whether to update memory (True during training)
        Returns:
            retrieved: Retrieved features from memory [B, C, H, W]
            attention_weights: Memory attention weights for analysis
        """
        B, C, H, W = features.shape
        assert C == self.feature_dim, f"Feature dim {C} != memory dim {self.feature_dim}"

        # Reshape features: [B, C, H, W] -> [B*HW, C]
        features_flat = features.permute(0, 2, 3, 1).contiguous()
        features_flat = features_flat.view(-1, C)

        # Project query
        query = self.query_proj(features_flat)
        query = F.normalize(query, dim=1)

        # Compute similarity with memory
        similarity = torch.matmul(query, self.memory.t())  # [B*HW, mem_size]

        # Hard shrinkage for sparsity
        if self.shrink_thres > 0:
            similarity = self.hard_shrink_relu(similarity, self.shrink_thres)

        # Soft attention over memory
        attention_weights = F.softmax(similarity * 10, dim=1)  # Temperature = 10

        # Retrieve from memory
        retrieved = torch.matmul(attention_weights, self.memory)  # [B*HW, C]

        # Update memory during training
        if update_memory and self.training:
            self.update(features_flat)

        # Reshape back
        retrieved = retrieved.view(B, H, W, C).permute(0, 3, 1, 2)

        return retrieved, attention_weights

    def hard_shrink_relu(self, x, threshold):
        """Hard shrinkage function for sparsity"""
        return F.relu(x - threshold)

    def update(self, features):
        """Update memory with exponential moving average"""
        with torch.no_grad():
            features_norm = F.normalize(features, dim=1)

            # Random sample for efficiency
            if features.size(0) > 100:
                indices = torch.randperm(features.size(0))[:100]
                features_norm = features_norm[indices]

            # Update memory with moving average
            for feat in features_norm:
                # Find most similar memory slot
                similarity = torch.matmul(feat.unsqueeze(0), self.memory.t())
                _, idx = torch.max(similarity, dim=1)

                # Update with moving average
                self.memory[idx] = F.normalize(
                    self.alpha * feat + (1 - self.alpha) * self.memory[idx],
                    dim=0
                )


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism that learns to focus on important spatial regions.
    Complements channel attention.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Spatially attended features [B, C, H, W]
        """
        # Compute spatial statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]

        # Concatenate
        spatial = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]

        # Compute attention map
        attention = self.sigmoid(self.conv(spatial))  # [B, 1, H, W]

        # Apply attention
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention sequentially.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttentionCBAM(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ChannelAttentionCBAM(nn.Module):
    """Channel attention for CBAM"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention
