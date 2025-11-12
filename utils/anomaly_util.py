import math
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F


def psnr_park(mse):
    return 10 * math.log10(1 / mse)


def anomaly_score(psnr, max_psnr, min_psnr):
    return (psnr - min_psnr) / (max_psnr - min_psnr)


def calculate_auc(config, psnr_list, mat):
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process

    scores = np.array([], dtype=np.float)
    labels = np.array([], dtype=np.int)

    for i in range(len(psnr_list)):
        score = anomaly_score(psnr_list[i], np.max(psnr_list[i]), np.min(psnr_list[i]))

        scores = np.concatenate((scores, score), axis=0)
        labels = np.concatenate((labels, mat[i][fp:]), axis=0)
    assert scores.shape == labels.shape, f'Ground truth has {labels.shape[0]} frames, BUT got {scores.shape[0]} detected frames!'
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr


class MultiScaleAnomalyScorer:
    """
    Enhanced anomaly scoring using multiple metrics for more robust detection
    """
    def __init__(self):
        self.weights = {
            'pixel': 0.3,
            'gradient': 0.25,
            'ssim': 0.25,
            'feature': 0.2
        }

    def compute_pixel_score(self, pred, target):
        """Pixel-level reconstruction error (MSE -> PSNR)"""
        mse = torch.mean((pred - target) ** 2).item()
        psnr = psnr_park(mse) if mse > 0 else 100
        # Convert PSNR to anomaly score (lower PSNR = higher anomaly)
        return 1.0 / (psnr + 1e-6)

    def compute_gradient_score(self, pred, target):
        """Gradient magnitude difference"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_x = sobel_x.to(pred.device)
        sobel_y = sobel_y.to(pred.device)

        # Compute gradients for each channel
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)

        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)

        grad_diff = torch.mean(torch.abs(pred_grad_x - target_grad_x) +
                              torch.abs(pred_grad_y - target_grad_y)).item()

        return grad_diff

    def compute_ssim_score(self, pred, target, window_size=11):
        """Structural similarity score"""
        # Simplified SSIM computation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        ssim_value = torch.mean(ssim_map).item()

        # Convert to anomaly score (lower SSIM = higher anomaly)
        return 1.0 - ssim_value

    def compute_feature_score(self, feat_pred, feat_target):
        """Feature-level distance (if encoder features are available)"""
        if feat_pred is None or feat_target is None:
            return 0.0

        feat_dist = torch.mean((feat_pred - feat_target) ** 2).item()
        return feat_dist

    def compute_anomaly_score(self, pred, target, feat_pred=None, feat_target=None):
        """
        Compute combined multi-scale anomaly score

        Args:
            pred: Predicted frame [B, C, H, W]
            target: Ground truth frame [B, C, H, W]
            feat_pred: Optional predicted features
            feat_target: Optional target features

        Returns:
            Combined anomaly score (higher = more anomalous)
        """
        pixel_score = self.compute_pixel_score(pred, target)
        gradient_score = self.compute_gradient_score(pred, target)
        ssim_score = self.compute_ssim_score(pred, target)
        feature_score = self.compute_feature_score(feat_pred, feat_target)

        # Weighted combination
        total_score = (
            self.weights['pixel'] * pixel_score +
            self.weights['gradient'] * gradient_score +
            self.weights['ssim'] * ssim_score +
            self.weights['feature'] * feature_score
        )

        return total_score

    def compute_psnr_for_compatibility(self, pred, target):
        """Compute PSNR for compatibility with existing evaluation code"""
        mse = torch.mean((pred - target) ** 2).item()
        return psnr_park(mse) if mse > 0 else 100


def calculate_comprehensive_metrics(scores, labels):
    """
    Calculate comprehensive evaluation metrics

    Args:
        scores: Anomaly scores
        labels: Ground truth labels (1=anomaly, 0=normal)

    Returns:
        Dictionary of metrics
    """
    # ROC-AUC
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # Average Precision (PR-AUC)
    ap = metrics.average_precision_score(labels, scores)

    # Equal Error Rate
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_idx]

    # F1 score at optimal threshold
    precision, recall, pr_thresholds = metrics.precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]

    return {
        'AUC': auc,
        'AP': ap,
        'EER': eer,
        'Best_F1': best_f1,
        'FPR': fpr,
        'TPR': tpr
    }