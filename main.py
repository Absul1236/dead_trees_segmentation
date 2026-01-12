import os
import glob
import numpy as np
from skimage import io, color, morphology
import matplotlib.pyplot as plt
import yaml


# =========================
# Configuration & IO helpers
# =========================

def get_project_root():
    """Return project root directory (script directory if available, otherwise current working directory)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def load_config_yaml(cfg_path):
    """Load configuration from a YAML file and return it as a Python dict."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_pngs(folder):
    """Return sorted list of all .png files inside a folder."""
    return sorted(glob.glob(os.path.join(folder, "*.png")))


def save_mask(mask, out_path):
    """Save a binary mask as PNG (values 0 or 255)."""
    m = (mask > 0).astype(np.uint8) * 255
    io.imsave(out_path, m)


# =========================
# Segmentation functions
# =========================

def generate_dead_tree_mask_nrg(nrg_img):
    """
    Generate dead-tree mask from NRG image using simple thresholding.
    Expected channel order: [NIR, R, G] or similar (uses channel 0 as NIR and 1 as Red).
    """
    nir = nrg_img[:, :, 0].astype(float)
    r = nrg_img[:, :, 1].astype(float)

    nir_max = nir.max() if nir.max() != 0 else 1.0
    r_max = r.max() if r.max() != 0 else 1.0

    nir /= nir_max
    r /= r_max

    mask = (nir < 0.32) & (r > 0.45)
    return mask.astype(np.uint8)


def dead_trees_mask_rgb_adaptive(
    rgb_image,
    h_low=0.74, h_high=0.05,
    s_q=60, v_q=25,
    min_s=0.08, min_v=0.08,
    max_v=1.00
):
    """
    Generate dead-tree mask from RGB image using HSV thresholds.
    Hue band targets purple/pink tones (wrap-around range). Saturation/value thresholds are computed adaptively.
    """
    hsv = color.rgb2hsv(rgb_image)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Wrap-around hue range for purple/pink tones
    hue_band = (H >= h_low) | (H <= h_high)

    # Adaptive thresholds inside the hue band (fallback to minimums if empty)
    if np.any(hue_band):
        S_band = S[hue_band]
        V_band = V[hue_band]
        s_thr = max(min_s, np.percentile(S_band, s_q))
        v_thr = max(min_v, np.percentile(V_band, v_q))
    else:
        s_thr = min_s
        v_thr = min_v

    mask = hue_band & (S >= s_thr) & (V >= v_thr) & (V <= max_v)
    return mask.astype(np.uint8)


def merge(mask_nrg, mask_rgb, radius=3):
    """
    Merge NRG and RGB predictions:
    - keep NRG positives
    - also keep RGB positives near NRG positives (using dilation neighborhood)
    """
    nrg = mask_nrg.astype(bool)
    rgb = mask_rgb.astype(bool)
    nrg_dil = morphology.binary_dilation(nrg, morphology.disk(radius))
    merged = nrg | (rgb & nrg_dil)
    return merged.astype(np.uint8)


def clean_mask_morph(mask, min_size=80, hole_size=120, radius=1):
    """
    Post-process a binary mask using morphological operations:
    - remove small objects
    - closing (connect nearby regions)
    - fill small holes
    - opening (remove noise)
    """
    mask_bool = (mask > 0)
    selem = morphology.disk(radius)

    clean = morphology.remove_small_objects(mask_bool, min_size=min_size)
    clean = morphology.binary_closing(clean, selem)
    clean = morphology.remove_small_holes(clean, area_threshold=hole_size)
    clean = morphology.binary_opening(clean, selem)

    return clean.astype(np.uint8)


def binarize_gt(gt_mask):
    """Convert a ground-truth mask to binary (0/1) using threshold >= 1."""
    return (gt_mask >= 1).astype(np.uint8)


# =========================
# Metrics & visualization
# =========================

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


# =========================
# Data loading (no validations as requested)
# =========================

def path_maker(input_mask_dir, input_rgb_dir, input_nrg_dir):
    """Return lists of mask/RGB/NRG PNG paths (no validation)."""
    paths_masks = list_pngs(input_mask_dir)
    paths_rgbs = list_pngs(input_rgb_dir)
    paths_nrgs = list_pngs(input_nrg_dir)
    return paths_masks, paths_rgbs, paths_nrgs


# =========================
# Main pipeline
# =========================

def main():
    """Run the full pipeline: load data, generate masks, save outputs, compute metrics, and plot results."""
    project_root = get_project_root()
    cfg = load_config_yaml(os.path.join(project_root, "config.yaml"))

    # Read globals from YAML
    iou_th = float(cfg["globals"]["iou_threshold"])
    print_limit = int(cfg["globals"]["print_limit"])
    all_limit = int(cfg["globals"]["all_limit"])

    # Read paths from YAML (relative to project root)
    input_rgb_dir = os.path.join(project_root, cfg["paths"]["rgb_dir"])
    input_nrg_dir = os.path.join(project_root, cfg["paths"]["nrg_dir"])
    input_mask_dir = os.path.join(project_root, cfg["paths"]["mask_dir"])

    results_dir = os.path.join(project_root, cfg["paths"]["results_dir"])
    out_rgb_dir = os.path.join(results_dir, cfg["paths"]["out_rgb_subdir"])
    out_nrg_dir = os.path.join(results_dir, cfg["paths"]["out_nrg_subdir"])
    out_merge_dir = os.path.join(results_dir, cfg["paths"]["out_merge_subdir"])

    # Create output directories
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_nrg_dir, exist_ok=True)
    os.makedirs(out_merge_dir, exist_ok=True)

    # Prepare data lists
    paths_masks, paths_rgbs, paths_nrgs = path_maker(input_mask_dir, input_rgb_dir, input_nrg_dir)
    max_n = min(all_limit, len(paths_masks))

    # Storage for statistics
    all_iou_nrg, all_iou_rgb, all_iou_merge = [], [], []
    tp_nrg, fp_nrg, fn_nrg, tn_nrg = [], [], [], []
    tp_rgb, fp_rgb, fn_rgb, tn_rgb = [], [], [], []
    tp_merge, fp_merge, fn_merge, tn_merge = [], [], [], []

    count_nrg = 0
    count_rgb = 0
    count_merge = 0
    limiter = 0
    operations_count = 0

    # Main loop
    for i in range(max_n):
        operations_count += 1

        # Ground truth
        gt = binarize_gt(io.imread(paths_masks[i]))
        gt_flat = gt.ravel().astype(np.uint8)

        # NRG prediction
        nrg_img = io.imread(paths_nrgs[i])
        pred_nrg = generate_dead_tree_mask_nrg(nrg_img)
        pred_nrg_flat = pred_nrg.ravel().astype(np.uint8)

        tpg, fpg, fng, tng = confusion_matrix(gt_flat, pred_nrg_flat)
        tp_nrg.append(tpg); fp_nrg.append(fpg); fn_nrg.append(fng); tn_nrg.append(tng)

        iou_nrg = compare_masks(gt, pred_nrg)
        all_iou_nrg.append(iou_nrg)

        # RGB prediction
        rgb_img = io.imread(paths_rgbs[i])
        pred_rgb = dead_trees_mask_rgb_adaptive(rgb_img)
        pred_rgb_flat = pred_rgb.ravel().astype(np.uint8)

        tpb, fpb, fnb, tnb = confusion_matrix(gt_flat, pred_rgb_flat)
        tp_rgb.append(tpb); fp_rgb.append(fpb); fn_rgb.append(fnb); tn_rgb.append(tnb)

        iou_rgb = compare_masks(gt, pred_rgb)
        all_iou_rgb.append(iou_rgb)

        # MERGE prediction
        pred_merge = merge(pred_nrg, pred_rgb)
        pred_merge = clean_mask_morph(pred_merge)
        pred_merge_flat = pred_merge.ravel().astype(np.uint8)

        tpm, fpm, fnm, tnm = confusion_matrix(gt_flat, pred_merge_flat)
        tp_merge.append(tpm); fp_merge.append(fpm); fn_merge.append(fnm); tn_merge.append(tnm)

        iou_merge = compare_masks(gt, pred_merge)
        all_iou_merge.append(iou_merge)

        # Save generated masks using RGB filename
        base_name = os.path.basename(paths_rgbs[i])
        save_mask(pred_nrg, os.path.join(out_nrg_dir, base_name))
        save_mask(pred_rgb, os.path.join(out_rgb_dir, base_name))
        save_mask(pred_merge, os.path.join(out_merge_dir, base_name))

        # Preview (limited)
        if limiter < print_limit:
            vis = combine_masks(gt, pred_rgb, pred_nrg)
            shower(
                i, paths_masks, paths_rgbs, paths_nrgs,
                iou_nrg, iou_rgb, iou_merge,
                pred_merge, vis
            )
            limiter += 1

        # Count above threshold
        if iou_nrg >= iou_th:
            count_nrg += 1
        if iou_rgb >= iou_th:
            count_rgb += 1
        if iou_merge >= iou_th:
            count_merge += 1

    # Summary output + plots
    print(f"Number of IoU >= {iou_th} | RGB: {count_rgb} | NRG: {count_nrg} | MERGE: {count_merge}")
    print("Processed images:", operations_count)
    print()

    print("NRG")
    plotter(all_iou_nrg, "NRG")
    plot_confusion_matrix(tp_nrg, fp_nrg, fn_nrg, tn_nrg, "NRG")
    print("TP:", sum(tp_nrg), "FP:", sum(fp_nrg), "FN:", sum(fn_nrg), "TN:", sum(tn_nrg))
    print()

    print("RGB")
    plotter(all_iou_rgb, "RGB")
    plot_confusion_matrix(tp_rgb, fp_rgb, fn_rgb, tn_rgb, "RGB")
    print("TP:", sum(tp_rgb), "FP:", sum(fp_rgb), "FN:", sum(fn_rgb), "TN:", sum(tn_rgb))
    print()

    print("MERGE")
    plotter(all_iou_merge, "MERGE")
    plot_confusion_matrix(tp_merge, fp_merge, fn_merge, tn_merge, "MERGE")
    sum_all = sum(tp_merge) + sum(fp_merge) + sum(fn_merge)
    sum_all = sum_all if sum_all != 0 else 1
    pr_tp = sum(tp_merge) / sum_all * 100
    pr_fp = sum(fp_merge) / sum_all * 100
    pr_fn = sum(fn_merge) / sum_all * 100
    print("TP:", sum(tp_merge), "FP:", sum(fp_merge), "FN:", sum(fn_merge), "TN:", sum(tn_merge))
    print("TP:", pr_tp, "%", "FP:", pr_fp, "%", "FN:", pr_fn, "%")


if __name__ == "__main__":
    main()