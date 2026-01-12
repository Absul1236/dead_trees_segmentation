import numpy as np
from skimage import color, morphology

"""
segmentation.py

Image segmentation algorithms.

This module contains all functions responsible for generating
binary masks of dead trees from:
- NRG images (threshold-based approach),
- RGB images (HSV-based adaptive approach),
- merged RGB + NRG predictions,
- morphological post-processing.

All parameters are provided externally (config or CLI),
making the algorithms fully configurable.
"""


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

    hue_band = (H >= h_low) | (H <= h_high)

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
