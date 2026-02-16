from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


API = "https://commons.wikimedia.org/w/api.php"

UA = "IPPR-GDD-StudentProject/0.1 (contact: local coursework; https://commons.wikimedia.org)"


ALLOWED_LICENSE_PREFIXES = (
    "CC BY",
    "CC BY-SA",
    "CC0",
    "Public domain",
    "PD",
)


@dataclass(frozen=True)
class CommonsItem:
    title: str
    page_url: str
    thumb_url: str
    thumb_width: int
    thumb_height: int
    license_short: str
    license_url: str
    artist: str
    credit: str


def _api_get(params: dict) -> dict:
    p = dict(params)
    p["format"] = "json"
    p["formatversion"] = 2
    r = requests.get(API, params=p, timeout=60, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json()


def _search_file_titles(query: str, limit: int) -> list[str]:
    titles: list[str] = []
    sroffset = 0
    while len(titles) < limit:
        batch = min(50, limit - len(titles))
        data = _api_get(
            {
                "action": "query",
                "list": "search",
                "srnamespace": 6,  # File:
                "srlimit": batch,
                "sroffset": sroffset,
                "srsearch": query,
            }
        )
        res = data.get("query", {}).get("search", [])
        if not res:
            break
        for it in res:
            t = str(it.get("title", ""))
            if t.startswith("File:"):
                titles.append(t)
        sroffset += len(res)
        if len(res) < batch:
            break
    # Dedup while preserving order
    seen = set()
    out = []
    for t in titles:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:limit]


def _category_file_titles(category: str, limit: int) -> list[str]:
    titles: list[str] = []
    cmcontinue = None
    while len(titles) < limit:
        batch = min(100, limit - len(titles))
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 6,  # File:
            "cmlimit": batch,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _api_get(params)
        res = data.get("query", {}).get("categorymembers", [])
        if not res:
            break
        for it in res:
            t = str(it.get("title", ""))
            if t.startswith("File:"):
                titles.append(t)
        cmcontinue = (data.get("continue") or {}).get("cmcontinue")
        if not cmcontinue:
            break

    seen = set()
    out = []
    for t in titles:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:limit]


def _extmeta_val(meta: dict, key: str) -> str:
    v = meta.get(key) or {}
    val = v.get("value")
    return "" if val is None else str(val)


def _fetch_item(title: str, thumb_width: int) -> CommonsItem | None:
    data = _api_get(
        {
            "action": "query",
            "prop": "imageinfo",
            "titles": title,
            "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": thumb_width,
        }
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    imageinfo = (page.get("imageinfo") or [])
    if not imageinfo:
        return None
    info = imageinfo[0]
    mime = str(info.get("mime") or "")
    # Skip non-image media like PDFs/DjVu (thumbs exist but aren't glove photos).
    if mime and not mime.startswith("image/"):
        return None
    # Also skip obvious document titles.
    lower = title.lower()
    if lower.endswith(".pdf") or lower.endswith(".djvu") or lower.endswith(".svg"):
        return None
    meta = info.get("extmetadata") or {}

    license_short = _extmeta_val(meta, "LicenseShortName")
    license_url = _extmeta_val(meta, "LicenseUrl")
    if not license_short:
        return None
    if not any(license_short.startswith(p) for p in ALLOWED_LICENSE_PREFIXES):
        return None

    thumb_url = str(info.get("thumburl") or info.get("url") or "")
    if not thumb_url:
        return None
    tw = int(info.get("thumbwidth") or 0)
    th = int(info.get("thumbheight") or 0)
    if tw <= 0 or th <= 0:
        # Some files may not have a thumb; fall back to original size if present.
        tw = int(info.get("width") or 0)
        th = int(info.get("height") or 0)

    artist = _extmeta_val(meta, "Artist")
    credit = _extmeta_val(meta, "Credit")
    page_url = f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"

    return CommonsItem(
        title=title,
        page_url=page_url,
        thumb_url=thumb_url,
        thumb_width=tw,
        thumb_height=th,
        license_short=license_short,
        license_url=license_url,
        artist=artist,
        credit=credit,
    )


def _safe_name(title: str) -> str:
    # "File:Something.jpg" -> "Something.jpg"
    name = title.split(":", 1)[-1]
    name = name.replace("/", "_").replace("\\", "_")
    return name


def _title_ok_for_type(glove_type: str, title: str) -> bool:
    """
    Commons search/categories can mix glove materials. Apply a strict heuristic filter
    based on the file title to reduce cross-contamination between classes.
    """
    t = title.lower()

    # Common cross-type words:
    exclude_all = ["condom", "mitt", "mitten"]
    if any(x in t for x in exclude_all):
        return False

    if glove_type == "latex":
        include = ["latex glove", "latex", "surgical glove", "medical glove"]
        exclude = ["nitrile", "vinyl", "polyethylene", "plastic", "leather glove", "work glove", "knit glove", "mitt", "mitten"]
        return any(x in t for x in include) and not any(x in t for x in exclude)

    if glove_type == "leather":
        include = ["leather glove", "leather gloves", "driving glove", "biker glove", "motorcycle glove", "riding glove"]
        exclude = ["nitrile", "latex", "vinyl glove", "polyethylene", "disposable glove", "disposable gloves", "knit glove"]
        return any(x in t for x in include) and not any(x in t for x in exclude)

    if glove_type == "fabric":
        include = ["knit glove", "knitted glove", "fabric glove", "winter glove", "work glove", "work gloves", "garden glove", "gardening glove", "glove"]
        exclude = ["nitrile", "latex", "vinyl glove", "polyethylene", "disposable glove", "disposable gloves", "leather glove"]
        return any(x in t for x in include) and not any(x in t for x in exclude)

    return True


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backoff = 1.5
    for attempt in range(6):
        r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": UA})
        if r.status_code == 429:
            # Rate limited by Commons; backoff and retry.
            time.sleep(backoff)
            backoff *= 1.8
            continue
        r.raise_for_status()
        break
    else:
        r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/public/commons_raw", help="Output folder")
    ap.add_argument("--thumb-width", type=int, default=1024, help="Download thumb width (reduces file sizes)")
    ap.add_argument("--per-type", type=int, default=60, help="Max items per glove type")
    ap.add_argument("--types", nargs="*", default=None, help="Subset of glove types to collect (default: all)")
    ap.add_argument("--use-categories", action="store_true", help="Also pull from relevant Commons categories")
    ap.add_argument("--sleep", type=float, default=0.35, help="Sleep seconds between downloads (reduces rate limiting)")
    ap.add_argument("--min-dim", type=int, default=500, help="Minimum thumbnail width/height to keep")
    ap.add_argument("--strict-title-filter", action="store_true", help="Apply strict title-based filtering per glove type")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Commons search syntax support can vary; use simple queries (multiple per type).
    queries: dict[str, list[str]] = {
        "latex": ["latex glove", "surgical latex glove", "medical latex glove", "latex examination glove"],
        "leather": ["leather glove", "driving glove", "motorcycle glove", "biker leather glove"],
        "fabric": ["knit glove", "knitted glove", "fabric glove", "winter glove"],
    }

    categories = {
        # Keep categories narrow to avoid cross-contamination between glove types.
        # Broad categories like "Disposable gloves" tend to mix latex/nitrile/vinyl/PE together.
        "latex": [],
        "leather": ["Leather gloves"],
        "fabric": ["Work gloves", "Garden gloves"],
    }

    glove_types = list(queries.keys())
    if args.types:
        wanted = [str(x) for x in args.types]
        bad = [x for x in wanted if x not in glove_types]
        if bad:
            raise SystemExit(f"Unknown glove types: {bad}. Valid: {glove_types}")
        glove_types = wanted

    manifest_rows: list[dict] = []
    for glove_type in glove_types:
        q_list = queries[glove_type]
        titles: list[str] = []
        for q in q_list:
            titles.extend(_search_file_titles(q, limit=int(args.per_type) * 4))
        if args.use_categories:
            for cat in categories.get(glove_type, []):
                titles.extend(_category_file_titles(cat, limit=int(args.per_type) * 3))
            # de-dup while preserving order
            seen = set()
            deduped = []
            for t in titles:
                if t in seen:
                    continue
                seen.add(t)
                deduped.append(t)
            titles = deduped
        items: list[CommonsItem] = []
        for t in titles:
            if args.strict_title_filter and not _title_ok_for_type(glove_type, t):
                continue
            it = _fetch_item(t, thumb_width=int(args.thumb_width))
            if it is None:
                continue
            # crude quality filter
            if it.thumb_width < int(args.min_dim) or it.thumb_height < int(args.min_dim):
                continue
            items.append(it)
            if len(items) >= int(args.per_type):
                break

        img_dir = out_dir / glove_type / "images"
        meta_dir = out_dir / glove_type / "meta"
        img_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        for it in tqdm(items, desc=f"Downloading {glove_type}", unit="img"):
            fname = _safe_name(it.title)
            out_path = img_dir / fname
            if not out_path.exists() or out_path.stat().st_size == 0:
                _download(it.thumb_url, out_path)
                time.sleep(float(args.sleep))
            (meta_dir / (fname + ".json")).write_text(json.dumps(it.__dict__, indent=2), encoding="utf-8")

            manifest_rows.append(
                {
                    "file": str(out_path.as_posix()),
                    "glove_type": glove_type,
                    "source": "Wikimedia Commons",
                    "title": it.title,
                    "page_url": it.page_url,
                    "license_short": it.license_short,
                    "license_url": it.license_url,
                    "artist": it.artist,
                    "credit": it.credit,
                }
            )

    import pandas as pd

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    readme = out_dir / "SOURCE.md"
    readme.write_text(
        "\n".join(
            [
                "# Wikimedia Commons data (public, open-licensed)",
                "",
                "These images are downloaded via the Wikimedia Commons API and filtered to Creative Commons / Public Domain licenses.",
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
