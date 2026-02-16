from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


API = "https://api.openverse.org/v1/images/"

UA = "IPPR-GDD-StudentProject/0.1 (contact: local coursework)"

# Openverse license short-codes:
# https://api.openverse.org/v1/#tag/images
ALLOWED_LICENSES_BASE = {
    "cc0",
    "pdm",  # public domain mark
    "by",
    "by-sa",
}


@dataclass(frozen=True)
class OVItem:
    id: str
    title: str
    url: str
    thumbnail: str
    width: int
    height: int
    license: str
    license_url: str
    creator: str
    creator_url: str
    source: str
    foreign_landing_url: str


def _api_get(params: dict) -> dict:
    backoff = 1.0
    last_exc: Exception | None = None
    for _ in range(6):
        try:
            r = requests.get(
                API,
                params=params,
                timeout=60,
                headers={
                    "User-Agent": UA,
                    "Accept": "application/json",
                },
            )
            if r.status_code in (401, 403, 429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 1.8
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - CLI script with retry/backoff
            last_exc = e
            time.sleep(backoff)
            backoff *= 1.8
    if last_exc:
        raise last_exc
    raise RuntimeError("Openverse API request failed")


def _safe_name(s: str) -> str:
    s = s.strip().replace("/", "_").replace("\\", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in (" ", "_", "-", ".", "(", ")")).strip()
    s = "_".join(s.split())
    return s[:180] if s else "image"


def _download(url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backoff = 1.2
    for attempt in range(6):
        r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": UA})
        # Some thumbnails can be temporarily unavailable or backed by providers
        # that fail dependencies (424). Treat those as non-fatal and skip.
        if r.status_code in (400, 401, 403, 404, 410, 424):
            return False
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff *= 1.7
            continue
        try:
            r.raise_for_status()
        except Exception:
            return False
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        return True
    return False


def _title_ok_for_type(glove_type: str, title: str) -> bool:
    t = (title or "").lower()
    if not t:
        return True
    if any(x in t for x in ["mitt", "mitten", "oven glove"]):
        return False
    if glove_type == "latex":
        return "latex" in t and "nitrile" not in t and "vinyl" not in t and "polyethylene" not in t
    if glove_type == "leather":
        return "leather" in t and "latex" not in t and "nitrile" not in t and "vinyl" not in t
    if glove_type == "fabric":
        return any(x in t for x in ["knit glove", "knitted glove", "fabric glove", "winter glove", "work glove", "garden glove", "gardening glove", "glove"]) and "leather" not in t
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/public/openverse_raw", help="Output folder")
    ap.add_argument("--per-type", type=int, default=120, help="Max items per glove type")
    ap.add_argument("--types", nargs="*", default=None, help="Subset of glove types to collect (default: all)")
    ap.add_argument("--page-size", type=int, default=20, help="API page size (anonymous limit is 20)")
    ap.add_argument("--min-dim", type=int, default=450, help="Minimum width/height to keep")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between downloads")
    ap.add_argument("--strict-title-filter", action="store_true", help="Apply strict title-based filtering per glove type")
    ap.add_argument(
        "--include-nc",
        action="store_true",
        help="Also allow NonCommercial licenses (by-nc, by-nc-sa). Only use if your submission/distribution is non-commercial.",
    )
    args = ap.parse_args()

    # Openverse enforces a strict anonymous limit.
    if int(args.page_size) > 20:
        args.page_size = 20

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries: dict[str, list[str]] = {
        "latex": ["latex glove", "surgical latex glove", "medical latex glove", "disposable latex glove"],
        "leather": ["leather glove", "leather gloves", "driving glove", "biker leather glove"],
        "fabric": ["knit glove", "knitted glove", "fabric glove", "winter glove"],
    }

    allowed_licenses = set(ALLOWED_LICENSES_BASE)
    if args.include_nc:
        allowed_licenses |= {"by-nc", "by-nc-sa"}

    allowed_license_param = ",".join(sorted(allowed_licenses))

    manifest_rows: list[dict] = []
    glove_types = list(queries.keys())
    if args.types:
        wanted = [str(x) for x in args.types]
        bad = [x for x in wanted if x not in glove_types]
        if bad:
            raise SystemExit(f"Unknown glove types: {bad}. Valid: {glove_types}")
        glove_types = wanted

    for glove_type in glove_types:
        q_list = queries[glove_type]
        items: list[OVItem] = []
        seen_ids: set[str] = set()
        pbar = tqdm(total=int(args.per_type), desc=f"Searching {glove_type}", unit="img")

        for q in q_list:
            page = 1
            while len(items) < int(args.per_type):
                data = _api_get(
                    {
                        "q": q,
                        "page": page,
                        "page_size": int(args.page_size),
                        "license": allowed_license_param,
                    }
                )
                res = data.get("results") or []
                if not res:
                    break
                for it in res:
                    oid = str(it.get("id") or "")
                    if not oid or oid in seen_ids:
                        continue
                    seen_ids.add(oid)

                    lic = str(it.get("license") or "").lower()
                    if lic not in allowed_licenses:
                        continue

                    # Mature content is irrelevant for gloves; skip it.
                    if bool(it.get("mature")):
                        continue

                    w = int(it.get("width") or 0)
                    h = int(it.get("height") or 0)
                    if w < int(args.min_dim) or h < int(args.min_dim):
                        continue

                    title = str(it.get("title") or "")
                    if args.strict_title_filter and not _title_ok_for_type(glove_type, title):
                        continue

                    thumb = str(it.get("thumbnail") or "")
                    url = str(it.get("url") or "")
                    if not thumb and not url:
                        continue

                    items.append(
                        OVItem(
                            id=oid,
                            title=title,
                            url=url,
                            thumbnail=thumb or url,
                            width=w,
                            height=h,
                            license=lic,
                            license_url=str(it.get("license_url") or ""),
                            creator=str(it.get("creator") or ""),
                            creator_url=str(it.get("creator_url") or ""),
                            source=str(it.get("source") or "openverse"),
                            foreign_landing_url=str(it.get("foreign_landing_url") or ""),
                        )
                    )
                    pbar.update(1)
                    if len(items) >= int(args.per_type):
                        break
                page += 1
                if not data.get("next"):
                    break
        pbar.close()

        img_dir = out_dir / glove_type / "images"
        meta_dir = out_dir / glove_type / "meta"
        img_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        for it in tqdm(items, desc=f"Downloading {glove_type}", unit="img"):
            fname = f"{glove_type}_{_safe_name(it.title) or it.id}_{it.id}.jpg"
            out_path = img_dir / fname
            if not out_path.exists() or out_path.stat().st_size == 0:
                ok = _download(it.thumbnail, out_path)
                if not ok:
                    # Skip broken/unavailable thumbnails.
                    if out_path.exists() and out_path.stat().st_size == 0:
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                    continue
                time.sleep(float(args.sleep))
            (meta_dir / (fname + ".json")).write_text(json.dumps(it.__dict__, indent=2), encoding="utf-8")

            manifest_rows.append(
                {
                    "file": out_path.as_posix(),
                    "glove_type": glove_type,
                    "source": "Openverse",
                    "title": it.title,
                    "page_url": it.foreign_landing_url,
                    "license_short": it.license,
                    "license_url": it.license_url,
                    "artist": it.creator,
                    "credit": it.creator_url,
                }
            )

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    (out_dir / "SOURCE.md").write_text(
        "\n".join(
            [
                "# Openverse data (public, open-licensed)",
                "",
                "These images are downloaded via the Openverse API and filtered to permissive licenses only (CC0, PDM, CC BY, CC BY-SA).",
                "Per-image attribution and license links are stored in `manifest.csv` and individual JSON files under each type's `meta/` folder.",
                "",
                "If you use any of these images in your report/submission, include proper citations and respect the license terms.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote manifest: {manifest_path}")
    print(f"Downloaded into: {out_dir}")


if __name__ == "__main__":
    main()
