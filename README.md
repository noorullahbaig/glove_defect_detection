# Glove Defect Detection

## Overview

Glove Defect Detection is a Python-based computer vision project for segmenting gloves, classifying glove type, and detecting visible defects from images. The repository combines a reusable analysis pipeline in `gdd/core/`, a Streamlit interface for inspection, and utility scripts for dataset setup, inference, training, and evaluation.

## Problem

The project focuses on identifying glove defects from images using a classical computer vision workflow rather than a deep-learning stack. The committed code and notes also make the project constraints explicit: no Haar cascades, no TensorFlow, and no template matching via `cv2.matchTemplate`.

## Approach

- Preprocess and normalize input images before segmentation
- Segment glove regions and compute derived image features
- Predict glove type with a dedicated classifier module
- Score structural and surface defect labels through the detection pipeline
- Expose the workflow through a Streamlit app and standalone scripts for inference and evaluation

## Repo Structure

- `gdd/core/` - core pipeline modules for preprocessing, segmentation, feature extraction, glove-type modeling, and defect detection
- `app/streamlit_app.py` - interactive Streamlit interface
- `scripts/` - utilities for dataset initialization, inference, evaluation, synthetic data setup, and threshold tuning
- `data/` - label files and dataset notes
- `datasets/` - dataset-specific documentation
- `report/` - report templates and project write-up support files

## How to Run or Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/init_dataset.py
streamlit run app/streamlit_app.py
```

For single-image inference, the repository also includes:

```bash
python scripts/run_infer.py <image-path> --out <output-overlay-path>
```

## Limitations

- Raw image datasets are not committed to the repository.
- Some workflows expect you to populate `data/labels.csv` and related image folders before training or evaluation.
- The glove-type model artifact is not guaranteed to be present, so glove type may remain unknown until a model is trained locally.
