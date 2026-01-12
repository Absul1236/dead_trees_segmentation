# Dead Tree Segmentation (RGB / NRG / MERGE)

This project implements a **modular Python pipeline** for segmentation of standing dead trees using aerial imagery.
The solution is based on **classical image processing techniques** (no machine learning) and is designed for
clarity, configurability, and reproducibility.

Three segmentation strategies are implemented and evaluated:

- **NRG-based segmentation** – thresholding on NIR and Red channels  
- **RGB-based segmentation** – adaptive thresholding in HSV color space  
- **MERGE segmentation** – combination of RGB and NRG predictions with spatial constraints  

The pipeline generates binary masks, saves results to disk, computes evaluation metrics,
and visualizes both qualitative and quantitative outputs.

---

## Dataset

The project uses an **external dataset** consisting of:
- RGB images  
- NRG images  
- ground-truth segmentation masks  

Dataset source (Kaggle):  
https://www.kaggle.com/datasets/meteahishali/aerial-imagery-for-standing-dead-tree-segmentation

The dataset is **not included** in this repository due to size and license restrictions.

---

## Project Structure

```
project/
├── main.py                  # main entry point (pipeline orchestration)
├── io_utils.py              # configuration, argparse, paths, IO helpers
├── segmentation.py          # segmentation algorithms (RGB / NRG / MERGE)
├── metrics.py               # IoU and confusion-matrix metrics
├── visualization.py         # plots and visual previews
├── summary_utils.py         # final reporting and summaries
│
├── copy_config.yaml         # configuration template (DO NOT EDIT)
├── config.yaml              # user configuration (created by user)
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── RGB_images/          # input RGB images (.png)
│   ├── NRG_images/          # input NRG images (.png)
│   └── masks/               # ground-truth masks (.png)
│
└── results/                 # generated outputs (created automatically)
    ├── RGB/
    ├── NRG/
    └── MERGE/
```

---

## Requirements

- Python **3.9 or newer**
- numpy
- scikit-image
- matplotlib
- pyyaml

---

## Configuration System

The project uses a two-stage configuration system:

- `copy_config.yaml` – read-only template  
- `config.yaml` – user-editable configuration  

The user must **manually create** `config.yaml` by copying the template:

```bash
cp copy_config.yaml config.yaml
```

All segmentation parameters, paths and runtime limits are defined in `config.yaml`.

---

## Running the Program

```bash
python main.py
```

Optional parameters can be overridden from the command line:

```bash
python main.py --all-limit 100 --iou-threshold 0.2
```

Command-line arguments always override values from `config.yaml`.

---

## Output

For each input image, three binary masks (0 / 255) are generated:

- `results/NRG/`
- `results/RGB/`
- `results/MERGE/`

The program also generates evaluation plots and summary statistics.

---

## Notes

- Input images must have matching filenames across folders.
- The project is intended for educational and experimental use.
