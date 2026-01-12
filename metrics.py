import numpy as np

"""
metrics.py

Evaluation metrics for segmentation results.

This module provides:
- ground-truth binarization,
- IoU (Intersection over Union) computation,
- confusion matrix calculation (TP, FP, FN, TN),
- conversion of raw counts to percentages.

No visualization or file I/O is performed here.
"""


def binarize_gt(gt_mask):
    """Convert a ground-truth mask to binary (0/1) using threshold >= 1."""
    return (gt_mask >= 1).astype(np.uint8)


def compare_masks(gt_mask, pred_mask):
    """Compute IoU (Intersection over Union) between two binary masks."""
    gt = (gt_mask > 0)
    pr = (pred_mask > 0)

    intersection = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    return float(intersection / union) if union != 0 else 0.0


def confusion_matrix(gt, pred):
    """Compute TP, FP, FN, TN from two binary 1D arrays (0/1)."""
    tp = ((gt == 1) & (pred == 1)).sum()
    fp = ((gt == 0) & (pred == 1)).sum()
    fn = ((gt == 1) & (pred == 0)).sum()
    tn = ((gt == 0) & (pred == 0)).sum()
    return int(tp), int(fp), int(fn), int(tn)


def print_percentages(tp, fp, fn, tn):
    """Convert TP/FP/FN/TN counts to percentages (0-100)."""
    total = tp + fp + fn + tn
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        tp / total * 100,
        fp / total * 100,
        fn / total * 100,
        tn / total * 100
    )
