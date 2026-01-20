# Dead Tree Segmentation (RGB / NRG / MERGE)

This project implements a Python-based pipeline for segmentation of standing dead trees using aerial imagery.
The pipeline is based on classical image processing methods (no machine learning) and is fully configurable
via a YAML configuration file and command-line arguments.

Three segmentation approaches are implemented:
- NRG-based segmentation (thresholding on NIR and Red channels),
- RGB-based segmentation (adaptive HSV thresholding),
- MERGE segmentation, combining RGB and NRG predictions.

The program generates binary masks, computes evaluation metrics, and saves reports and plots to files
(no on-screen visualization by default).

---

## Dataset

The project uses an external dataset consisting of:
- RGB images,
- NRG images,
- ground-truth segmentation masks.

Dataset source (Kaggle):
https://www.kaggle.com/datasets/meteahishali/aerial-imagery-for-standing-dead-tree-segmentation

The dataset is not included in this repository due to size and license restrictions.

---

## Project Structure

project/
├── main.py                  # main executable script
├── io_utils.py              # configuration, CLI, logging, IO helpers
├── segmentation.py          # segmentation algorithms
├── metrics.py               # evaluation metrics
├── visualization.py         # plot generation (file-only)
├── summary_utils.py         # reporting utilities
│
├── config_copy.yaml         # configuration template (DO NOT EDIT)
├── config.yaml              # user configuration (created by the user)
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── examples_rgb/
│   ├── examples_nrg/
│   └── examples_gt_masks/
│
└── results/
    └── reports/

---

## Requirements

- Python 3.9+
- numpy
- scikit-image
- matplotlib
- pyyaml

---

## Virtual Environment Setup

Create a virtual environment:

python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Linux / macOS:
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

## Configuration (config.yaml)

Before running the program, you must create config.yaml by copying the template:

cp config_copy.yaml config.yaml

Do not edit config_copy.yaml. Only modify config.yaml.

All paths and global parameters are defined in config.yaml, including:
- input image directories,
- output directory for generated masks,
- reports directory for logs and plots,
- segmentation thresholds and morphological parameters.

---

## Running the Program

Run the pipeline from the project root:

python main.py

During execution, the program:
1. Loads configuration (YAML + CLI overrides),
2. Reads RGB, NRG and ground-truth images,
3. Generates segmentation masks,
4. Saves all generated masks to disk,
5. Computes IoU and confusion-matrix statistics,
6. Saves logs and plots to files.

By default, the program does not print results to the terminal.
All textual output is written to a log file, and plots are saved to a report image file.

To also mirror logs to the console, run:

python main.py --console

---

## Output

For each input image, the following files are produced:
- RGB-based segmentation mask,
- NRG-based segmentation mask,
- MERGE (RGB + NRG) segmentation mask.

Additionally, the program saves:
- results/reports/run.log – full textual output,
- results/reports/metrics_report.png – combined IoU and TP/FP/FN plots.

---

## Notes

- Input images must have matching filenames across modalities.
- All paths may be absolute or relative.
- The pipeline is designed for reproducible batch processing.
