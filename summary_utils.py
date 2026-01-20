from metrics import print_percentages
from visualization import save_metrics_report

"""
summary_utils.py

Final summary and reporting utilities (file-only mode).

This module:
- logs summary values to a logger (typically configured for file output),
- saves one combined plots figure to disk (report file).
"""


def summary(
    iou_th,
    count_rgb, count_nrg, count_merge,
    operations_count,
    all_iou_nrg, all_iou_rgb, all_iou_merge,
    tp_nrg, fp_nrg, fn_nrg, tn_nrg,
    tp_rgb, fp_rgb, fn_rgb, tn_rgb,
    tp_merge, fp_merge, fn_merge, tn_merge,
    report_file,
    logger
):
    """
    File-only reporting:
    - text output is written via logger
    - plots are saved to a single file (report_file)
    """
    logger.info(f"IoU >= {iou_th} | RGB: {count_rgb} | NRG: {count_nrg} | MERGE: {count_merge}")
    logger.info(f"Processed images: {operations_count}")

    # NRG
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_nrg), sum(fp_nrg), sum(fn_nrg), sum(tn_nrg))
    logger.info(f"NRG  | TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")

    # RGB
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_rgb), sum(fp_rgb), sum(fn_rgb), sum(tn_rgb))
    logger.info(f"RGB  | TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")

    # MERGE
    tp_p, fp_p, fn_p, tn_p = print_percentages(sum(tp_merge), sum(fp_merge), sum(fn_merge), sum(tn_merge))
    logger.info(f"MERGE| TP: {tp_p:.2f}% | FP: {fp_p:.2f}% | FN: {fn_p:.2f}% | TN: {tn_p:.2f}%")

    # Save ONE combined figure report
    save_metrics_report(
        out_path=report_file,
        all_iou_nrg=all_iou_nrg, all_iou_rgb=all_iou_rgb, all_iou_merge=all_iou_merge,
        tp_nrg=tp_nrg, fp_nrg=fp_nrg, fn_nrg=fn_nrg, tn_nrg=tn_nrg,
        tp_rgb=tp_rgb, fp_rgb=fp_rgb, fn_rgb=fn_rgb, tn_rgb=tn_rgb,
        tp_merge=tp_merge, fp_merge=fp_merge, fn_merge=fn_merge, tn_merge=tn_merge
    )
    logger.info(f"Saved plots report: {report_file}")
