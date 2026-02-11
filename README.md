# IPPR Assignment — Glove Defect Detection (GDD)

This repository contains a **classical computer vision** prototype for the IPPR in-course group assignment:
**segment the glove**, **classify glove type**, and **detect + name defects** — without using **Haar cascades**, **TensorFlow**, or **template matching**.

## Quick start

1) Create a virtual environment and install dependencies:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If dependency installation fails (commonly due to OpenCV wheels), use **Python 3.11 or 3.12**.

2) Initialize the dataset folders + labels template:

```bash
python scripts/init_dataset.py
```

3) Run the GUI (Streamlit):

```bash
streamlit run app/streamlit_app.py
```

## Trial end-to-end demo (downloads a small public image set)

This is only to prove the pipeline runs end-to-end quickly.

```bash
python scripts/download_trial_dataset.py
python scripts/run_infer.py data/public/trial_glove_defect/images/s1.jpg --out results/overlays/s1_overlay.png
python scripts/evaluate.py --labels data/trial_labels.csv
streamlit run app/streamlit_app.py
```

In the GUI you can upload any of the downloaded images from `data/public/trial_glove_defect/images/`.

## What to expect (trial run)

- If `gdd/models/glove_type.joblib` does not exist yet, glove type will show as `unknown`.
  - Train it once you have your own labeled images: `python scripts/train_glove_type.py`
- Defect detections are heuristics and may produce extra boxes on the trial images.
  - The goal of the trial run is **end-to-end execution** (segmentation → defect overlay → GUI), not final accuracy.

## Project layout

- `gdd/`: core pipeline (preprocess, segmentation, features, defect detection)
- `app/`: Streamlit GUI
- `scripts/`: dataset utilities and evaluation scripts
- `data/`: dataset structure (raw images go here; not committed)
- `report/`: report outline/template and tables/figures outputs

## Data expectations (labels.csv)

Your main labels file is `data/labels.csv` with (at minimum) these columns:

- `file`: relative path to image (from repo root)
- `glove_type`: one of `nitrile`, `plastic`, `fabric`
- `defect_labels`: pipe-separated labels (multi-label), e.g. `spotting|stain_dirty`
- `split`: `train`, `val`, or `test`
- `lighting`: free text (e.g. `daylight`, `warm_indoor`, `dim_indoor`)
- `background`: free text (e.g. `plain_desk`, `patterned_desk`, `colored_mat`)
- `notes`: free text

If you label defect regions (recommended), add:
- `bboxes`: JSON list of `{label,x,y,w,h}` in pixel coordinates

## Defect label list (recommended)

This repo uses these canonical labels (edit in `gdd/core/labels.py` if you need to rename):

- `discoloration`
- `spotting`
- `stain_dirty`
- `plastic_contamination`
- `tear`
- `hole`
- `wrinkles_dent`
- `damaged_by_fold`
- `inside_out`
- `improper_roll`
- `incomplete_beading`
- `missing_finger`

## Notes on constraints

- No `cv2.matchTemplate` (template matching).
- No Haar cascade usage.
- No TensorFlow usage.
