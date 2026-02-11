# GUI Guide (Streamlit)

## Start the app

```bash
source .venv_gdd/bin/activate
streamlit run app/streamlit_app.py
```

If you used a different virtualenv, activate it instead.

## Single Image tab

1) Upload a `.jpg`/`.png` image.
2) The app runs:
   - preprocessing (illumination normalization + denoise)
   - glove segmentation (mask overlay)
   - defect detection (red boxes + defect table)
   - glove type (if a trained model is available)

Outputs shown:
- Left: original image
- Right: segmentation overlay + defect boxes
- Below: predicted glove type + list of defects and scores

## Batch Folder tab

1) Set `Folder path (local)` to something like:
   - `data/raw` (your captured images)
   - `data/public/trial_glove_defect/images` (trial dataset)
2) Click **Run batch**

Files written:
- `results/overlays/*_overlay.png` (visual outputs)
- `results/batch_results.csv` (predictions per file)

## Common “why is glove type unknown?”

The pipeline loads the glove-type classifier from:
- `gdd/models/glove_type.joblib`

Train it after you have a real dataset in `data/labels.csv`:

```bash
python scripts/train_glove_type.py --labels data/labels.csv --out gdd/models/glove_type.joblib
```

