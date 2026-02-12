# Trial dataset (for demo only)

This repo includes a **downloader script** for a small set of glove images so you can run an end-to-end demo quickly.

## How to download

```bash
.venv_gdd/bin/python scripts/download_trial_dataset.py
```

It downloads images into `data/public/trial_glove_defect/images/` and writes `data/trial_labels.csv`.

## Quick run commands

```bash
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/python scripts/run_infer.py data/public/trial_glove_defect/images/s1.jpg --out results/overlays/s1_overlay.png
.venv_gdd/bin/python scripts/evaluate.py --labels data/trial_labels.csv
```

To inspect through GUI, start Streamlit with:

```bash
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/streamlit run app/streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```

## Source

The script downloads from the GitHub repository:

`avishkakavindu/defect-detection-opencv-python`

## Important

- This is for a **trial run** of the pipeline + GUI.
- For your coursework submission, you should use **your own captured dataset** (and/or properly licensed datasets) and cite sources in APA.
