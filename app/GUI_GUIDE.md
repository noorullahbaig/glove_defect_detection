# GUI Guide (Streamlit)

## Start the app

```bash
cd "/Users/ayaanminhas/Desktop/IPPR Assignment"
.venv_gdd/bin/streamlit run app/streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```

Keep this terminal window open while you are using the app.

Expected output includes:
- Local URL: `http://localhost:8501`
- Network URL: `http://<your-ip>:8501`

If you get Streamlit connection error / refused connection:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
```

If no process is listening, restart with the exact command above.

## Compare Uploads tab

1) Upload one or more `.jpg`/`.png` images and click **Add uploads**.
2) The app runs:
   - preprocessing (illumination normalization + denoise)
   - glove segmentation (mask overlay)
   - defect detection (boxes when available)
   - glove type (if a trained model is available)

Outputs shown:
- Each uploaded image stays visible for side-by-side comparison.
- A comparison table shows predicted glove type and defect scores.

Default score filters:
- Structural labels: min score `0.65`
- Surface/color labels: min score `0.85`

Defect dropdown:
- Select **All** to run and display all defects.
- Select a specific defect to run that defect only and show a single score for it.

## Batch Folder tab

1) Set `Folder path (local)` to something like:
   - `data/raw` (your captured images)
   - `data/public/trial_glove_defect/images` (trial dataset)
   - `data/my_test` (curated defect sanity checks)
2) Click **Run batch**

Files written:
- `results/overlays/*_overlay.png` (visual outputs)
- `results/batch_results.csv` (predictions per file)

## Curated test set (quick verification)

Use `data/my_test` to verify behavior quickly:
- `missing_finger_4_fingers_and_one_cut_finger.png` -> expected `missing_finger`
- `hole_on_fingertip.png` -> expected `hole`
- `hole_in_the_middle.png` -> expected `hole`
- stain/discoloration files -> expected `discoloration`

## Common “why is glove type unknown?”

The pipeline loads the glove-type classifier from:
- `gdd/models/glove_type.joblib`

Train it after you have a real dataset in `data/labels.csv`:

```bash
.venv_gdd/bin/python scripts/train_glove_type.py --labels data/labels.csv --out gdd/models/glove_type.joblib
```
