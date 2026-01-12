import numpy as np
import matplotlib.pyplot as plt
from skimage import io

"""
visualization.py

Visualization utilities.

This module is responsible for:
- plotting IoU histograms,
- plotting confusion matrix distributions,
- creating RGB/NRG/GT overlay images,
- displaying per-image previews during processing.

All functions here are optional for analysis and debugging
and do not affect segmentation results.
"""

def plot_confusion_matrix(tp, fp, fn, tn, name):
    """Plot TP/FP/FN proportions for a method."""
    labels = ["TP", "FP", "FN"]
    values = [sum(tp), sum(fp), sum(fn)]
    total = sum(values) if sum(values) != 0 else 1
    values = [sum(tp) / total * 100, sum(fp) / total * 100, sum(fn) / total * 100]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(f"Confusion Matrix values for {name}")
    plt.xlabel("Category")
    plt.ylabel("Count in %")
    plt.show()


def plotter(all_iou, name):
    """Plot histogram of IoU scores and show basic statistics."""
    plt.figure(figsize=(10, 6))
    plt.hist(all_iou, bins=20, edgecolor="black", alpha=0.7)
    plt.title(f"Distribution of IoU Scores for {name}")
    plt.xlabel("IoU Score")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.75)

    stats_text = (
        f"Mean IoU: {np.mean(all_iou):.4f}\n"
        f"Median IoU: {np.median(all_iou):.4f}\n"
        f"Min IoU: {np.min(all_iou):.4f}\n"
        f"Max IoU: {np.max(all_iou):.4f}"
    )
    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
    )
    plt.show()


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


def shower(idx, paths_masks, paths_rgbs, paths_nrgs, iou_nrg, iou_rgb, iou_merge, merged_mask, rgb_merge_vis):
    """Show preview figure for one sample (RGB, NRG, GT, overlay, merged) and print IoU values."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 5))

    axes[0].imshow(io.imread(paths_rgbs[idx]))
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(io.imread(paths_nrgs[idx]))
    axes[1].set_title("NRG")
    axes[1].axis("off")

    axes[2].imshow(io.imread(paths_masks[idx]), cmap="gray")
    axes[2].set_title("Ground truth")
    axes[2].axis("off")

    axes[3].imshow(rgb_merge_vis)
    axes[3].set_title("NRG/RGB/GT (R/G/B)")
    axes[3].axis("off")

    axes[4].imshow(merged_mask, cmap="gray")
    axes[4].set_title("MERGE")
    axes[4].axis("off")

    print(f"IoU_NRG: {iou_nrg:.4f} | IoU_RGB: {iou_rgb:.4f} | IoU_MERGE: {iou_merge:.4f}")
    plt.show()
