# Dead Tree Segmentation (RGB / NRG / MERGE)

This project implements a **file-only** Python pipeline for segmentation of standing dead trees using aerial imagery.
The pipeline is based on classical image processing methods (no machine learning) and is fully configurable
via a YAML configuration file and command-line arguments.

Three segmentation approaches are implemented:
- **NRG-based segmentation** (thresholding on NIR and Red channels),
- **RGB-based segmentation** (adaptive HSV thresholding),
- **MERGE segmentation**, combining RGB and NRG predictions.

The program generates binary masks, computes evaluation metrics, and **saves all results to user-defined locations on disk**.
Nothing is displayed on screen by default. Logs are written to a file and can optionally be mirrored to the console.

---

## Dataset

The project uses an external dataset consisting of:
- RGB images,
- NRG images,
- ground-truth segmentation masks.

Dataset source (Kaggle):  
https://www.kaggle.com/datasets/meteahishali/aerial-imagery-for-standing-dead-tree-segmentation

The dataset is **not included** in this repository due to size and license restrictions.

---

## Project Structure

```
project/
├── main.py
├── io_utils.py
├── segmentation.py
├── metrics.py
├── visualization.py
├── summary_utils.py
│
├── config_copy.yaml
├── config.yaml
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
```

---

## Requirements

- Python 3.9+
- numpy
- scikit-image
- matplotlib
- pyyaml

---

## Configuration System

### Template file

`config_copy.yaml` is a template file and **must not be edited**.

### User configuration (REQUIRED)

Before running the program, the user must create `config.yaml` by copying:

```bash
cp config_copy.yaml config.yaml
```

Only `config.yaml` should be modified by the user.

---

## User-Defined Output Paths

All output paths are defined by the user in `config.yaml` or overridden via CLI.

Example:

```yaml
paths:
  results_dir: "results"
  out_rgb_subdir: "RGB"
  out_nrg_subdir: "NRG"
  out_merge_subdir: "MERGE"
  reports_subdir: "reports"
```

The program automatically creates all required directories.

---

## Running the Program

```bash
python main.py
```

The pipeline:
1. Loads configuration (YAML + CLI overrides),
2. Reads RGB, NRG and ground-truth images,
3. Generates segmentation masks,
4. Saves masks to user-defined directories,
5. Computes evaluation metrics,
6. Saves logs and plots to user-defined report locations.

No GUI windows are displayed. Console output is optional and controlled via a CLI flag.

---

## Command-Line Overrides

Any configuration value can be overridden via CLI:

```bash
python main.py --results-dir D:/my_results --all-limit 100
```

Enable console logging in addition to file logging:
```bash
python main.py --console
```

CLI arguments always override `config.yaml`.

---

## Output

- Binary masks (RGB / NRG / MERGE)
- `run.log` – full textual log
- `metrics_report.png` – combined evaluation plots

All outputs are saved to user-defined directories.

---

## Notes

- Input images must have matching filenames across modalities.
- Paths may be relative or absolute.
- Designed for reproducible batch processing.
