import os
import glob
import argparse
import logging
import yaml
import numpy as np
from skimage import io

"""
io_utils.py

Input / Output and configuration utilities.

This module handles:
- project root detection,
- loading YAML configuration files,
- parsing command-line arguments (argparse),
- applying CLI overrides on top of config.yaml,
- basic filesystem operations (listing images, saving masks),
- project-wide logging configuration (file-first, no console by default).
"""


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


def path_maker(input_mask_dir, input_rgb_dir, input_nrg_dir):
    """Return lists of mask/RGB/NRG PNG paths (no validation)."""
    paths_masks = list_pngs(input_mask_dir)
    paths_rgbs = list_pngs(input_rgb_dir)
    paths_nrgs = list_pngs(input_nrg_dir)
    return paths_masks, paths_rgbs, paths_nrgs


def save_mask(mask, out_path):
    """Save a binary mask as PNG (values 0 or 255)."""
    m = (mask > 0).astype(np.uint8) * 255
    io.imsave(out_path, m)


# =========================
# Logging
# =========================

def setup_logger(level="INFO", log_file=None, console=False):
    """
    Configure and return a project-wide logger.

    - level: DEBUG/INFO/WARNING/ERROR/CRITICAL
    - log_file: path to a .log file (recommended for file-only mode)
    - console: if False -> no console output; logs go only to file
    """
    logger = logging.getLogger("dead_tree_segmentation")

    # Clear handlers to avoid duplicates (important in interactive runs)
    logger.handlers.clear()

    level_upper = str(level).upper()
    logger.setLevel(getattr(logging, level_upper, logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if (not console) and (not log_file):
        raise ValueError("File-only logging is enabled (console=False), but no log_file was provided.")

    return logger


# =========================
# CLI / overrides
# =========================

def parse_args():
    """
    Parse command-line arguments.

    Every parameter defined in config.yaml is also available as a CLI override.
    CLI arguments override config.yaml only when provided (non-None).
    """
    parser = argparse.ArgumentParser(
        description="Dead tree segmentation pipeline (RGB / NRG / MERGE)."
    )

    # ---- globals ----
    parser.add_argument("--iou-threshold", type=float, default=None, help="Override globals.iou_threshold")
    parser.add_argument("--all-limit", type=int, default=None, help="Override globals.all_limit")
    parser.add_argument("--print-limit", type=int, default=None, help="Override globals.print_limit (not used in file-only mode)")

    # ---- paths ----
    parser.add_argument("--rgb-dir", type=str, default=None, help="Override paths.rgb_dir")
    parser.add_argument("--nrg-dir", type=str, default=None, help="Override paths.nrg_dir")
    parser.add_argument("--mask-dir", type=str, default=None, help="Override paths.mask_dir")
    parser.add_argument("--results-dir", type=str, default=None, help="Override paths.results_dir")
    parser.add_argument("--out-rgb-subdir", type=str, default=None, help="Override paths.out_rgb_subdir")
    parser.add_argument("--out-nrg-subdir", type=str, default=None, help="Override paths.out_nrg_subdir")
    parser.add_argument("--out-merge-subdir", type=str, default=None, help="Override paths.out_merge_subdir")
    parser.add_argument("--reports-subdir", type=str, default=None, help="Override paths.reports_subdir")

    # ---- logging ----
    parser.add_argument("--log-level", type=str, default=None, help="Override logging.level (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--log-filename", type=str, default=None, help="Override logging.filename (saved inside reports folder)")
    parser.add_argument("--console", action="store_true", help="Enable console logging (default: file-only)")

    # ---- report ----
    parser.add_argument("--report-filename", type=str, default=None, help="Override report.filename (saved inside reports folder)")

    # ---- mask generation: NRG ----
    parser.add_argument("--nrg-nir-threshold", type=float, default=None, help="Override mask_generation.nrg.nir_threshold")
    parser.add_argument("--nrg-r-threshold", type=float, default=None, help="Override mask_generation.nrg.r_threshold")

    # ---- mask generation: RGB ----
    parser.add_argument("--rgb-h-low", type=float, default=None, help="Override mask_generation.rgb.h_low")
    parser.add_argument("--rgb-h-high", type=float, default=None, help="Override mask_generation.rgb.h_high")
    parser.add_argument("--rgb-s-q", type=float, default=None, help="Override mask_generation.rgb.s_q")
    parser.add_argument("--rgb-v-q", type=float, default=None, help="Override mask_generation.rgb.v_q")
    parser.add_argument("--rgb-min-s", type=float, default=None, help="Override mask_generation.rgb.min_s")
    parser.add_argument("--rgb-min-v", type=float, default=None, help="Override mask_generation.rgb.min_v")
    parser.add_argument("--rgb-max-v", type=float, default=None, help="Override mask_generation.rgb.max_v")

    # ---- mask generation: MERGE ----
    parser.add_argument("--merge-radius", type=int, default=None, help="Override mask_generation.merge.radius")

    # ---- mask generation: CLEAN ----
    parser.add_argument("--clean-min-size", type=int, default=None, help="Override mask_generation.clean.min_size")
    parser.add_argument("--clean-hole-size", type=int, default=None, help="Override mask_generation.clean.hole_size")
    parser.add_argument("--clean-radius", type=int, default=None, help="Override mask_generation.clean.radius")

    return parser.parse_args()


def apply_cli_overrides(cfg, args, project_root):
    """
    Apply CLI overrides to config values and return resolved runtime values.

    Returns:
    - iou_th, print_limit, all_limit
    - input_rgb_dir, input_nrg_dir, input_mask_dir
    - results_dir, out_rgb_dir, out_nrg_dir, out_merge_dir
    - reports_dir, log_level, log_file, report_file
    - nrg_cfg, rgb_cfg, merge_cfg, clean_cfg
    """
    # --- globals ---
    iou_th = float(cfg["globals"]["iou_threshold"])
    print_limit = int(cfg["globals"]["print_limit"])
    all_limit = int(cfg["globals"]["all_limit"])

    if args.iou_threshold is not None:
        iou_th = args.iou_threshold
    if args.print_limit is not None:
        print_limit = args.print_limit
    if args.all_limit is not None:
        all_limit = args.all_limit

    # --- paths (relative to project root by default) ---
    input_rgb_dir = os.path.join(project_root, cfg["paths"]["rgb_dir"])
    input_nrg_dir = os.path.join(project_root, cfg["paths"]["nrg_dir"])
    input_mask_dir = os.path.join(project_root, cfg["paths"]["mask_dir"])

    results_dir = os.path.join(project_root, cfg["paths"]["results_dir"])
    out_rgb_subdir = cfg["paths"]["out_rgb_subdir"]
    out_nrg_subdir = cfg["paths"]["out_nrg_subdir"]
    out_merge_subdir = cfg["paths"]["out_merge_subdir"]
    reports_subdir = cfg["paths"]["reports_subdir"]

    # CLI overrides for input/output paths
    if args.rgb_dir is not None:
        input_rgb_dir = args.rgb_dir
    if args.nrg_dir is not None:
        input_nrg_dir = args.nrg_dir
    if args.mask_dir is not None:
        input_mask_dir = args.mask_dir
    if args.results_dir is not None:
        results_dir = args.results_dir

    if args.out_rgb_subdir is not None:
        out_rgb_subdir = args.out_rgb_subdir
    if args.out_nrg_subdir is not None:
        out_nrg_subdir = args.out_nrg_subdir
    if args.out_merge_subdir is not None:
        out_merge_subdir = args.out_merge_subdir
    if args.reports_subdir is not None:
        reports_subdir = args.reports_subdir

    out_rgb_dir = os.path.join(results_dir, out_rgb_subdir)
    out_nrg_dir = os.path.join(results_dir, out_nrg_subdir)
    out_merge_dir = os.path.join(results_dir, out_merge_subdir)
    reports_dir = os.path.join(results_dir, reports_subdir)

    # --- logging ---
    log_level = str(cfg.get("logging", {}).get("level", "INFO"))
    log_filename = str(cfg.get("logging", {}).get("filename", "run.log"))

    if args.log_level is not None:
        log_level = args.log_level
    if args.log_filename is not None:
        log_filename = args.log_filename

    log_file = os.path.join(reports_dir, log_filename)

    # --- report ---
    report_filename = str(cfg.get("report", {}).get("filename", "metrics_report.png"))
    if args.report_filename is not None:
        report_filename = args.report_filename
    report_file = os.path.join(reports_dir, report_filename)

    # --- mask generation parameters ---
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

    # MERGE overrides
    if args.merge_radius is not None:
        merge_cfg["radius"] = args.merge_radius

    # CLEAN overrides
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
        reports_dir, log_level, log_file, report_file,
        nrg_cfg, rgb_cfg, merge_cfg, clean_cfg
    )
