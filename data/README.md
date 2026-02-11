# Data folder

This repository expects images and labels to live under `data/`, but **raw images should not be committed**.

Run:

```bash
python scripts/init_dataset.py
```

Then place your captured images under:

- `data/raw/nitrile/<defect_label>/...`
- `data/raw/latex/<defect_label>/...`
- `data/raw/fabric/<defect_label>/...`

And fill in `data/labels.csv`.

