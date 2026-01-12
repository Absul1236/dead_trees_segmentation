import os
import numpy as np
from skimage import io

"""
main.py

Main entry point of the project.
Responsible for:
- loading configuration (YAML + CLI overrides),
- orchestrating the full processing pipeline,
- calling segmentation, evaluation and visualization modules.

This file does NOT contain algorithmic logic.
It only connects individual modules into a single workflow.
"""

from io_utils import (
    get_project_root,
    load_config_yaml,
    parse_args,
    apply_cli_overrides,
    save_mask,
    path_maker,
)
from segmentation import (
    generate_dead_tree_mask_nrg,
    dead_trees_mask_rgb_adaptive,
    merge,
    clean_mask_morph,
)
from metrics import (
    binarize_gt,
    compare_masks,
    confusion_matrix,
)
from visualization import (
    combine_masks,
    shower,
)
from summary_utils import summary


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
