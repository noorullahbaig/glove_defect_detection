# AI-Generated Dataset (Optional)

This folder is for AI-generated glove images used to test the pipeline and (optionally) train basic ML models.

## Layout

- Images (do not commit):
  - `data/ai_generated/images/<glove_type>/<defect_label>/...`
  - Use `normal` as the defect folder for clean gloves (no defect).
- Labels CSV (commit):
  - `data/ai_generated/labels.csv`

`<glove_type>` must be one of:
- `latex`
- `leather`
- `fabric`

`<defect_label>` should be one of the defect labels in `gdd/core/labels.py` or `normal`.

## Initialize / regenerate labels

Use:

```bash
.venv_gdd/bin/python scripts/init_ai_generated_dataset.py --root data/ai_generated
```

If images already exist under `data/ai_generated/images`, the script will scan folders and write `data/ai_generated/labels.csv` with a train/val/test split.

