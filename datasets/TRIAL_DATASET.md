# Trial dataset (for demo only)

This repo includes a **downloader script** for a small set of glove images so you can run an end-to-end demo quickly.

## How to download

```bash
python scripts/download_trial_dataset.py
```

It downloads images into `data/public/trial_glove_defect/images/` and writes `data/trial_labels.csv`.

## Source

The script downloads from the GitHub repository:

`avishkakavindu/defect-detection-opencv-python`

## Important

- This is for a **trial run** of the pipeline + GUI.
- For your coursework submission, you should use **your own captured dataset** (and/or properly licensed datasets) and cite sources in APA.

