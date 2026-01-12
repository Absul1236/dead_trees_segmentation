import os
import glob
import argparse
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


def parse_args():
    """
    Parse command-line arguments.
    Every parameter from config.yaml can be overridden via CLI.
    """
    parser = argparse.ArgumentParser(
        description="Dead tree segmentation pipeline using RGB, NRG and MERGE approaches."
    )

    # =========================
    # Globals
    # =========================
    parser.add_argument("--iou-threshold", type=float, default=None, help="Override globals.iou_threshold")
    parser.add_argument("--print-limit", type=int, default=None, help="Override globals.print_limit")
    parser.add_argument("--all-limit", type=int, default=None, help="Override globals.all_limit")

    # =========================
    # Paths
    # =========================
    parser.add_argument("--rgb-dir", type=str, default=None, help="Override paths.rgb_dir (input RGB dir)")
    parser.add_argument("--nrg-dir", type=str, default=None, help="Override paths.nrg_dir (input NRG dir)")
    parser.add_argument("--mask-dir", type=str, default=None, help="Override paths.mask_dir (input masks dir)")
    parser.add_argument("--results-dir", type=str, default=None, help="Override paths.results_dir (output root dir)")

    parser.add_argument("--out-rgb-subdir", type=str, default=None, help="Override paths.out_rgb_subdir")
    parser.add_argument("--out-nrg-subdir", type=str, default=None, help="Override paths.out_nrg_subdir")
    parser.add_argument("--out-merge-subdir", type=str, default=None, help="Override paths.out_merge_subdir")

    # =========================
    # NRG mask generation
    # =========================
    parser.add_argument("--nrg-nir-threshold", type=float, default=None, help="Override mask_generation.nrg.nir_threshold")
    parser.add_argument("--nrg-r-threshold", type=float, default=None, help="Override mask_generation.nrg.r_threshold")

    # =========================
    # RGB mask generation
    # =========================
    parser.add_argument("--rgb-h-low", type=float, default=None, help="Override mask_generation.rgb.h_low")
    parser.add_argument("--rgb-h-high", type=float, default=None, help="Override mask_generation.rgb.h_high")
    parser.add_argument("--rgb-s-q", type=float, default=None, help="Override mask_generation.rgb.s_q")
    parser.add_argument("--rgb-v-q", type=float, default=None, help="Override mask_generation.rgb.v_q")
    parser.add_argument("--rgb-min-s", type=float, default=None, help="Override mask_generation.rgb.min_s")
    parser.add_argument("--rgb-min-v", type=float, default=None, help="Override mask_generation.rgb.min_v")
    parser.add_argument("--rgb-max-v", type=float, default=None, help="Override mask_generation.rgb.max_v")

    # =========================
    # Merge parameters
    # =========================
    parser.add_argument("--merge-radius", type=int, default=None, help="Override mask_generation.merge.radius")

    # =========================
    # Morphological cleaning
    # =========================
    parser.add_argument("--clean-min-size", type=int, default=None, help="Override mask_generation.clean.min_size")
    parser.add_argument("--clean-hole-size", type=int, default=None, help="Override mask_generation.clean.hole_size")
    parser.add_argument("--clean-radius", type=int, default=None, help="Override mask_generation.clean.radius")

    return parser.parse_args()


def apply_cli_overrides(cfg, args, project_root):
    """
    Apply CLI overrides to config values.

    Returns resolved runtime values:
    - iou_th, print_limit, all_limit
    - input_rgb_dir, input_nrg_dir, input_mask_dir
    - results_dir, out_rgb_dir, out_nrg_dir, out_merge_dir
    - nrg_cfg, rgb_cfg, merge_cfg, clean_cfg (mask generation parameters)
    """

    # --- globals (from YAML) ---
    iou_th = float(cfg["globals"]["iou_threshold"])
    print_limit = int(cfg["globals"]["print_limit"])
    all_limit = int(cfg["globals"]["all_limit"])

    if args.iou_threshold is not None:
        iou_th = args.iou_threshold
    if args.print_limit is not None:
        print_limit = args.print_limit
    if args.all_limit is not None:
        all_limit = args.all_limit

    # --- paths (from YAML, relative to project root) ---
    input_rgb_dir = os.path.join(project_root, cfg["paths"]["rgb_dir"])
    input_nrg_dir = os.path.join(project_root, cfg["paths"]["nrg_dir"])
    input_mask_dir = os.path.join(project_root, cfg["paths"]["mask_dir"])

    results_dir = os.path.join(project_root, cfg["paths"]["results_dir"])

    # Override paths from CLI if provided (treated as user-provided paths)
    if args.rgb_dir is not None:
        input_rgb_dir = args.rgb_dir
    if args.nrg_dir is not None:
        input_nrg_dir = args.nrg_dir
    if args.mask_dir is not None:
        input_mask_dir = args.mask_dir
    if args.results_dir is not None:
        results_dir = args.results_dir

    # Output subdirs (default from YAML)
    out_rgb_subdir = cfg["paths"]["out_rgb_subdir"]
    out_nrg_subdir = cfg["paths"]["out_nrg_subdir"]
    out_merge_subdir = cfg["paths"]["out_merge_subdir"]

    # Override output subdir names from CLI
    if args.out_rgb_subdir is not None:
        out_rgb_subdir = args.out_rgb_subdir
    if args.out_nrg_subdir is not None:
        out_nrg_subdir = args.out_nrg_subdir
    if args.out_merge_subdir is not None:
        out_merge_subdir = args.out_merge_subdir

    out_rgb_dir = os.path.join(results_dir, out_rgb_subdir)
    out_nrg_dir = os.path.join(results_dir, out_nrg_subdir)
    out_merge_dir = os.path.join(results_dir, out_merge_subdir)

    # --- mask generation parameters (copied from YAML so we can override safely) ---
    nrg_cfg = dict(cfg["mask_generation"]["nrg"])
    rgb_cfg = dict(cfg["mask_generation"]["rgb"])
    merge_cfg = dict(cfg["mask_generation"]["merge"])
    clean_cfg = dict(cfg["mask_generation"]["clean"])

    # NRG overrides
    if args.nrg_nir_threshold is not None:
        nrg_cfg["nir_threshold"] = args.nrg_nir_threshold
    if args.nrg_r_threshold is not None:
        nrg_cfg["r_threshold"] = args.nrg_r_threshold

    # RGB overrides
    if args.rgb_h_low is not None:
        rgb_cfg["h_low"] = args.rgb_h_low
    if args.rgb_h_high is not None:
        rgb_cfg["h_high"] = args.rgb_h_high
    if args.rgb_s_q is not None:
        rgb_cfg["s_q"] = args.rgb_s_q
    if args.rgb_v_q is not None:
        rgb_cfg["v_q"] = args.rgb_v_q
    if args.rgb_min_s is not None:
        rgb_cfg["min_s"] = args.rgb_min_s
    if args.rgb_min_v is not None:
        rgb_cfg["min_v"] = args.rgb_min_v
    if args.rgb_max_v is not None:
        rgb_cfg["max_v"] = args.rgb_max_v

    # Merge overrides
    if args.merge_radius is not None:
        merge_cfg["radius"] = args.merge_radius

    # Clean overrides
    if args.clean_min_size is not None:
        clean_cfg["min_size"] = args.clean_min_size
    if args.clean_hole_size is not None:
        clean_cfg["hole_size"] = args.clean_hole_size
    if args.clean_radius is not None:
        clean_cfg["radius"] = args.clean_radius

    return (
        iou_th, print_limit, all_limit,
        input_rgb_dir, input_nrg_dir, input_mask_dir,
        results_dir, out_rgb_dir, out_nrg_dir, out_merge_dir,
        nrg_cfg, rgb_cfg, merge_cfg, clean_cfg
    )


# =========================
# Segmentation functions
# =========================

def generate_dead_tree_mask_nrg(nrg_img, nir_thr=0.32, r_thr=0.45):
    """
    Generate dead-tree mask from NRG image using thresholding.
    Uses channel 0 as NIR and channel 1 as Red. Thresholds are configurable.
    """
    nir = nrg_img[:, :, 0].astype(float)
    r = nrg_img[:, :, 1].astype(float)

    nir_max = nir.max() if nir.max() != 0 else 1.0
    r_max = r.max() if r.max() != 0 else 1.0

    nir /= nir_max
    r /= r_max

    mask = (nir < nir_thr) & (r > r_thr)
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
# Data loading
# =========================

def path_maker(input_mask_dir, input_rgb_dir, input_nrg_dir):
    """Return lists of mask/RGB/NRG PNG paths (no validation)."""
    paths_masks = list_pngs(input_mask_dir)
    paths_rgbs = list_pngs(input_rgb_dir)
    paths_nrgs = list_pngs(input_nrg_dir)
    return paths_masks, paths_rgbs, paths_nrgs


# =========================
# Summary
# =========================

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


def summary(iou_th,
            count_rgb, count_nrg, count_merge,
            operations_count,
            all_iou_nrg, all_iou_rgb, all_iou_merge,
            tp_nrg, fp_nrg, fn_nrg, tn_nrg,
            tp_rgb, fp_rgb, fn_rgb, tn_rgb,
            tp_merge, fp_merge, fn_merge, tn_merge):
    """
    Print summary statistics and display plots for NRG, RGB and MERGE methods.
    All TP / FP / FN / TN values are reported as percentages with 2 decimal places.
    """
    print(f"Number of IoU >= {iou_th} | RGB: {count_rgb} | NRG: {count_nrg} | MERGE: {count_merge}")
    print("Processed images:", operations_count)
    print()

    # -------- NRG --------
    print("NRG")
    plotter(all_iou_nrg, "NRG")
    plot_confusion_matrix(tp_nrg, fp_nrg, fn_nrg, tn_nrg, "NRG")
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_nrg), sum(fp_nrg), sum(fn_nrg), sum(tn_nrg))
    print(f"TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")
    print()

    # -------- RGB --------
    print("RGB")
    plotter(all_iou_rgb, "RGB")
    plot_confusion_matrix(tp_rgb, fp_rgb, fn_rgb, tn_rgb, "RGB")
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_rgb), sum(fp_rgb), sum(fn_rgb), sum(tn_rgb))
    print(f"TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")
    print()

    # -------- MERGE --------
    print("MERGE")
    plotter(all_iou_merge, "MERGE")
    plot_confusion_matrix(tp_merge, fp_merge, fn_merge, tn_merge, "MERGE")
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_merge), sum(fp_merge), sum(fn_merge), sum(tn_merge))
    print(f"TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")


# =========================
# Main pipeline
# =========================

def main():
    """Run the full pipeline: load data, generate masks, save outputs, compute metrics, and plot results."""
    args = parse_args()
    project_root = get_project_root()
    cfg = load_config_yaml(os.path.join(project_root, "config.yaml"))

    (
        iou_th, print_limit, all_limit,
        input_rgb_dir, input_nrg_dir, input_mask_dir,
        results_dir, out_rgb_dir, out_nrg_dir, out_merge_dir,
        nrg_cfg, rgb_cfg, merge_cfg, clean_cfg
    ) = apply_cli_overrides(cfg, args, project_root)

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
        pred_nrg = generate_dead_tree_mask_nrg(
            nrg_img,
            nir_thr=float(nrg_cfg["nir_threshold"]),
            r_thr=float(nrg_cfg["r_threshold"])
        )
        pred_nrg_flat = pred_nrg.ravel().astype(np.uint8)

        tpg, fpg, fng, tng = confusion_matrix(gt_flat, pred_nrg_flat)
        tp_nrg.append(tpg); fp_nrg.append(fpg); fn_nrg.append(fng); tn_nrg.append(tng)

        iou_nrg = compare_masks(gt, pred_nrg)
        all_iou_nrg.append(iou_nrg)

        # RGB prediction
        rgb_img = io.imread(paths_rgbs[i])
        pred_rgb = dead_trees_mask_rgb_adaptive(
            rgb_img,
            h_low=float(rgb_cfg["h_low"]),
            h_high=float(rgb_cfg["h_high"]),
            s_q=float(rgb_cfg["s_q"]),
            v_q=float(rgb_cfg["v_q"]),
            min_s=float(rgb_cfg["min_s"]),
            min_v=float(rgb_cfg["min_v"]),
            max_v=float(rgb_cfg["max_v"])
        )
        pred_rgb_flat = pred_rgb.ravel().astype(np.uint8)

        tpb, fpb, fnb, tnb = confusion_matrix(gt_flat, pred_rgb_flat)
        tp_rgb.append(tpb); fp_rgb.append(fpb); fn_rgb.append(fnb); tn_rgb.append(tnb)

        iou_rgb = compare_masks(gt, pred_rgb)
        all_iou_rgb.append(iou_rgb)

        # MERGE prediction
        pred_merge = merge(pred_nrg, pred_rgb, radius=int(merge_cfg["radius"]))
        pred_merge = clean_mask_morph(
            pred_merge,
            min_size=int(clean_cfg["min_size"]),
            hole_size=int(clean_cfg["hole_size"]),
            radius=int(clean_cfg["radius"])
        )
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
    summary(
        iou_th,
        count_rgb, count_nrg, count_merge,
        operations_count,
        all_iou_nrg, all_iou_rgb, all_iou_merge,
        tp_nrg, fp_nrg, fn_nrg, tn_nrg,
        tp_rgb, fp_rgb, fn_rgb, tn_rgb,
        tp_merge, fp_merge, fn_merge, tn_merge
    )


if __name__ == "__main__":
    main()
