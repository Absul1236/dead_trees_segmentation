from visualization import plotter, plot_confusion_matrix
from metrics import print_percentages

"""
summary_utils.py

Final summary and reporting utilities.

This module aggregates results from the entire dataset and:
- prints final statistics,
- reports TP / FP / FN / TN as percentages,
- calls visualization functions for final plots.

It separates reporting logic from the main pipeline.
"""


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

    # NRG
    print("NRG")
    plotter(all_iou_nrg, "NRG")
    plot_confusion_matrix(tp_nrg, fp_nrg, fn_nrg, tn_nrg, "NRG")
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_nrg), sum(fp_nrg), sum(fn_nrg), sum(tn_nrg))
    print(f"TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")
    print()

    # RGB
    print("RGB")
    plotter(all_iou_rgb, "RGB")
    plot_confusion_matrix(tp_rgb, fp_rgb, fn_rgb, tn_rgb, "RGB")
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_rgb), sum(fp_rgb), sum(fn_rgb), sum(tn_rgb))
    print(f"TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")
    print()

    # MERGE
    print("MERGE")
    plotter(all_iou_merge, "MERGE")
    plot_confusion_matrix(tp_merge, fp_merge, fn_merge, tn_merge, "MERGE")
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_merge), sum(fp_merge), sum(fn_merge), sum(tn_merge))
    print(f"TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")
