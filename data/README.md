# Data folder

This repository expects images and labels to live under `data/`, but **raw images should not be committed**.

Run:

```bash
.venv_gdd/bin/python scripts/init_dataset.py
```

Then place your captured images under:

- `data/raw/nitrile/<defect_label>/...`
- `data/raw/plastic/<defect_label>/...`
- `data/raw/fabric/<defect_label>/...`

And fill in `data/labels.csv`.

## Curated manual test set

Use `data/my_test/` as a quick functional validation set for focus-mode defects.

Current files include:
- `missing_finger_4_fingers_and_one_cut_finger.png`
- `hole_on_fingertip.png`
- `hole_in_the_middle.png`
- `stain:discoloration_in_the_middle_and_on_fingertips.png`
- `stain:discolouration_in_the_middle.png`
- `cut_middle_finger_exposing_skin.png`

Note:
- filenames are treated as intended defect hints for fast manual checking
- this folder is for sanity checks, not formal model evaluation
