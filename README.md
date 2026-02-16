# IPPR Assignment - Glove Defect Detection (GDD)

This repository contains a classical computer vision pipeline for glove analysis:
- segment glove area
- classify glove type
- detect and label glove defects

Constraints followed in this project:
- no Haar cascades
- no TensorFlow
- no template matching (`cv2.matchTemplate`)

## Current status (updated: February 12, 2026)

Progress in focus mode (`missing_finger`, `extra_fingers`, `hole`, `discoloration`, `damaged_by_fold`):
- missing-finger logic improved for 4-vs-3 finger-count deficit cases
- fingertip cut/shortening conflict is now handled as a structural defect
- curated tests in `data/my_test` now detect:
  - `missing_finger_4_fingers_and_one_cut_finger.png` -> `missing_finger`
  - `hole_on_fingertip.png` -> `hole`
  - `hole_in_the_middle.png` -> `hole`
  - stain/discoloration samples -> `discoloration`

Known limitation:
- `cut_middle_finger_exposing_skin.png` can still be classified mainly as `discoloration` in focus mode, depending on segmentation and threshold settings.

## Environment setup

Use Python 3.11 or 3.12 if wheel installation fails.

```bash
python3 --version
python3 -m venv .venv_gdd
source .venv_gdd/bin/activate
pip install -r requirements.txt
```

Initialize dataset folders/template:

```bash
python scripts/init_dataset.py
```

## Run the app (recommended and verified method)

Always start Streamlit from repo root with:

```bash
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/streamlit run app/streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```

Expected startup output includes:
- Local URL: `http://localhost:8501`
- Network URL: `http://<your-ip>:8501`

Important:
- keep this terminal session open while using the app
- if the process stops, the browser shows Streamlit connection error

## Streamlit connection troubleshooting

If you see `ERR_CONNECTION_REFUSED` or Streamlit "Connection error":

1. Check if port 8501 is listening:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
```

2. If nothing is listening, restart with the exact run command above.

3. If an old/stuck process exists, stop and relaunch:

```bash
pkill -f "streamlit run app/streamlit_app.py"
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/streamlit run app/streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```

## GUI behavior and thresholds

In `app/streamlit_app.py`:
- structural min score default: `0.65`
- surface/color min score default: `0.85`
- defect dropdown supports either:
  - `All` (run all defects)
  - a single selected defect (run only that defect)

Structural labels include:
- `hole`
- `tear`
- `missing_finger`
- `extra_fingers`
- `inside_out`
- `improper_roll`
- `incomplete_beading`
- `damaged_by_fold`

Surface labels include:
- `discoloration`
- `stain_dirty`
- `spotting`
- `plastic_contamination`

## Curated test set (`data/my_test`)

Current curated files:
- `data/my_test/cut_middle_finger_exposing_skin.png`
- `data/my_test/hole_in_the_middle.png`
- `data/my_test/hole_on_fingertip.png`
- `data/my_test/missing_finger_4_fingers_and_one_cut_finger.png`
- `data/my_test/stain:discoloration_in_the_middle_and_on_fingertips.png`
- `data/my_test/stain:discolouration_in_the_middle.png`

Run quick inference on one image:

```bash
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/python scripts/run_infer.py data/my_test/missing_finger_4_fingers_and_one_cut_finger.png --out results/overlays/my_test_missing_overlay.png
```

## Trial/demo commands

```bash
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/python scripts/download_trial_dataset.py
.venv_gdd/bin/python scripts/run_infer.py data/public/trial_glove_defect/images/s1.jpg --out results/overlays/s1_overlay.png
.venv_gdd/bin/python scripts/evaluate.py --labels data/trial_labels.csv
```

## Glove type model

If `gdd/models/glove_type.joblib` is missing, glove type may appear as `unknown`.

Train model:

```bash
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/python scripts/train_glove_type.py --labels data/labels.csv --out gdd/models/glove_type.joblib
```

## Data expectations (`data/labels.csv`)

Minimum columns:
- `file`
- `glove_type` (`latex`, `leather`, `fabric`)
- `defect_labels` (pipe-separated multi-label string)
- `split` (`train`, `val`, `test`)
- `lighting`
- `background`
- `notes`

Optional column:
- `bboxes` JSON list of `{label, x, y, w, h}`

## Defect label list (canonical)

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
- `extra_fingers`

## Project layout

- `gdd/` core pipeline (preprocess, segmentation, features, detectors)
- `app/` Streamlit app
- `scripts/` utilities (data prep, training, evaluation, inference)
- `data/` datasets and test images
- `results/` overlays, evaluation outputs
- `report/` assignment report materials
