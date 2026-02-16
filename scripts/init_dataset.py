from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdd.core.labels import DEFECT_LABELS, GLOVE_TYPES


def main() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    for gt in GLOVE_TYPES:
        for dl in DEFECT_LABELS + ["normal"]:
            Path(f"data/raw/{gt}/{dl}").mkdir(parents=True, exist_ok=True)

    labels_path = Path("data/labels.csv")
    if not labels_path.exists():
        df = pd.DataFrame(
            [
                {
                    "file": "data/raw/latex/normal/IMG_0001.jpg",
                    "glove_type": "latex",
                    "defect_labels": "",
                    "split": "train",
                    "lighting": "daylight",
                    "background": "plain_desk",
                    "notes": "",
                }
            ]
        )
        df.to_csv(labels_path, index=False)
        print(f"Created {labels_path}")
    else:
        print(f"Already exists: {labels_path}")

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/overlays").mkdir(parents=True, exist_ok=True)
    Path("gdd/models").mkdir(parents=True, exist_ok=True)
    Path("data/ai_generated/images").mkdir(parents=True, exist_ok=True)
    print("Dataset folders initialized.")


if __name__ == "__main__":
    main()
