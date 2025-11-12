import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


class VideoAugmentation:
    """
    Comprehensive video augmentation including spatial and temporal transforms
    for improved model robustness and generalization.
    """
    def __init__(self, config, train=True):
        self.train = train
        self.config = config

        # Spatial augmentations
        self.spatial_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                  saturation=0.2, hue=0.1),
        ])

        self.flip_prob = 0.5
        self.temporal_flip_prob = 0.3

    def __call__(self, frames):
        """
        Apply augmentation to a list of PIL Image frames

        Args:
            frames: List of PIL Images
        Returns:
            Augmented list of PIL Images
        """
        if not self.train:
            return frames

        # Apply temporal augmentations
        frames = self.temporal_flip(frames)

        # Apply spatial augmentations consistently across all frames
        frames = self.apply_spatial_augmentations(frames)

        return frames

    def apply_spatial_augmentations(self, frames):
        """Apply same spatial augmentation to all frames for temporal consistency"""
        # Random horizontal flip
        if random.random() < self.flip_prob:
            frames = [TF.hflip(frame) for frame in frames]

        # Apply color jitter with same parameters to all frames
        color_jitter_params = self.spatial_transforms.transforms[0].get_params(
            self.spatial_transforms.transforms[0].brightness,
            self.spatial_transforms.transforms[0].contrast,
            self.spatial_transforms.transforms[0].saturation,
            self.spatial_transforms.transforms[0].hue
        )

        frames = [TF.adjust_brightness(frame, color_jitter_params[0]) for frame in frames]
        frames = [TF.adjust_contrast(frame, color_jitter_params[1]) for frame in frames]
        frames = [TF.adjust_saturation(frame, color_jitter_params[2]) for frame in frames]
        frames = [TF.adjust_hue(frame, color_jitter_params[3]) for frame in frames]

        return frames

    def temporal_flip(self, frames):
        """Randomly reverse video sequence"""
        if random.random() < self.temporal_flip_prob:
            return list(reversed(frames))
        return frames


class RandomTemporalSampling:
    """Sample frames with varying temporal intervals"""
    def __init__(self, max_interval=2):
        self.max_interval = max_interval

    def __call__(self, frames):
        if random.random() < 0.3:  # 30% chance
            interval = random.randint(1, self.max_interval)
            if len(frames) > interval:
                return frames[::interval]
        return frames
