import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # disable GUI backends 
import matplotlib.pyplot as plt

from skimage import io

"""
visualization.py

Visualization utilities.

This module is responsible for:
- creating overlay images (optional),
- saving a single combined report figure with plots (file-only mode).

No GUI windows are shown. All figures are saved to disk.
"""


def combine_masks(original_mask, rgb_mask, nrg_mask):
    """
    Create a 3-channel visualization image:
    - Red: NRG prediction
    - Green: RGB prediction
    - Blue: ground truth
    """
    original = (original_mask > 0)
    nrg = (nrg_mask > 0)
    rgb = (rgb_mask > 0)

    nrg_channel = (nrg * 255).astype(np.uint8)
    rgb_channel = (rgb * 255).astype(np.uint8)
    original_channel = (original * 255).astype(np.uint8)

    combined = np.stack([nrg_channel, rgb_channel, original_channel], axis=-1)
    return combined.astype(np.uint8)


def save_metrics_report(
    out_path,
    all_iou_nrg, all_iou_rgb, all_iou_merge,
    tp_nrg, fp_nrg, fn_nrg, tn_nrg,
    tp_rgb, fp_rgb, fn_rgb, tn_rgb,
    tp_merge, fp_merge, fn_merge, tn_merge
):
    """
    Save ONE figure containing:
    - IoU histograms for NRG/RGB/MERGE
    - TP/FP/FN percentage bars for NRG/RGB/MERGE
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Row 1: IoU histograms ---
    axes[0, 0].hist(all_iou_nrg, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("NRG IoU distribution")
    axes[0, 0].set_xlabel("IoU")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(all_iou_rgb, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("RGB IoU distribution")
    axes[0, 1].set_xlabel("IoU")
    axes[0, 1].set_ylabel("Count")

    axes[0, 2].hist(all_iou_merge, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 2].set_title("MERGE IoU distribution")
    axes[0, 2].set_xlabel("IoU")
    axes[0, 2].set_ylabel("Count")

    # --- Row 2: TP/FP/FN in % (TN omitted for readability in bars) ---
    def _tp_fp_fn_percent(tp, fp, fn):
        tp_s, fp_s, fn_s = sum(tp), sum(fp), sum(fn)
        total = tp_s + fp_s + fn_s
        if total == 0:
            return [0.0, 0.0, 0.0]
        return [tp_s / total * 100, fp_s / total * 100, fn_s / total * 100]

    labels = ["TP", "FP", "FN"]

    axes[1, 0].bar(labels, _tp_fp_fn_percent(tp_nrg, fp_nrg, fn_nrg))
    axes[1, 0].set_title("NRG TP/FP/FN (%)")
    axes[1, 0].set_ylabel("%")

    axes[1, 1].bar(labels, _tp_fp_fn_percent(tp_rgb, fp_rgb, fn_rgb))
    axes[1, 1].set_title("RGB TP/FP/FN (%)")
    axes[1, 1].set_ylabel("%")

    axes[1, 2].bar(labels, _tp_fp_fn_percent(tp_merge, fp_merge, fn_merge))
    axes[1, 2].set_title("MERGE TP/FP/FN (%)")
    axes[1, 2].set_ylabel("%")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
