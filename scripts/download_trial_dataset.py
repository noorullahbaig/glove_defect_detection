from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


REPO = "avishkakavindu/defect-detection-opencv-python"
FILES = [
    "s1.jpg",
    "s2.JPG",
    "s3.JPG",
    "s4.JPG",
    "s5.JPG",
    "s6.JPG",
    "s7.JPG",
    "s8.JPG",
    "s9.JPG",
    "s10.JPG",
    "t1.jpg",
    "x1.jpeg",
    "x2.jpeg",
    "x3.jpeg",
    "x4.jpeg",
    "x5.jpeg",
]


def _download_one(out_path: Path, url: str) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return False
    out_path.write_bytes(r.content)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/public/trial_glove_defect", help="Folder to download into")
    ap.add_argument("--labels-out", default="data/trial_labels.csv", help="Write a trial labels CSV here")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split value for trial rows")
    ap.add_argument(
        "--defect-label",
        default="spotting",
        help="Defect label to assign to all trial images (demo only).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Try both common default branches for robustness.
    branches = ["main", "master"]

    downloaded = []
    for name in FILES:
        out_path = img_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            downloaded.append(out_path)
            continue

        ok = False
        for br in branches:
            url = f"https://raw.githubusercontent.com/{REPO}/{br}/images/{name}"
            if _download_one(out_path, url):
                ok = True
                break
        if not ok:
            raise SystemExit(f"Failed to download {name} from {REPO} (tried branches: {branches})")
        downloaded.append(out_path)

    # Trial labels are intentionally simple (single glove type + one defect class).
    # This is for an end-to-end demo only; for your report you should create your own labeled dataset.
    rows = []
    for p in downloaded:
        rows.append(
            {
                "file": str(p.as_posix()),
                "glove_type": "latex",
                "defect_labels": str(args.defect_label),
                "split": args.split,
                "lighting": "unknown",
                "background": "unknown",
                "notes": f"trial download from {REPO}",
            }
        )

    labels_out = Path(args.labels_out)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(labels_out, index=False)

    source_md = out_dir / "SOURCE.md"
    if not source_md.exists():
        source_md.write_text(
            "\n".join(
                [
                    "# Trial dataset source",
                    "",
                    "These images are downloaded for a quick end-to-end demo of the GDD pipeline.",
                    "",
                    f"Source repository: https://github.com/{REPO}",
                    "",
                    "Before using any public images in your final submission/report, verify license/permissions and cite appropriately.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    print(f"Downloaded {len(downloaded)} images to: {img_dir}")
    print(f"Wrote trial labels CSV: {labels_out}")


if __name__ == "__main__":
    main()
