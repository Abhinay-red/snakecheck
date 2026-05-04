"""
dataset_download.py

Downloads snake images from iNaturalist for SnakeCheck Hyderabad.
Fetches the Big Four venomous species + non-venomous snakes,
filtered to India geographic bounding box (lat 12-20N, lon 76-82E).

Usage:
    python scripts/dataset_download.py
    python scripts/dataset_download.py --max_per_class 500   # quick test run
    python scripts/dataset_download.py --resume               # skip already-downloaded
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INATURALIST_API = "https://api.inaturalist.org/v1"

# Taxon IDs verified against iNaturalist API (PRD §17.2 had stale IDs)
VENOMOUS_CLASSES = {
    "cobra": {
        "taxon_id": 53492,   # Naja naja
        "label": "cobra",
        "telugu": "నాగుపాము",
        "target": 1500,
    },
    "russells_viper": {
        "taxon_id": 30687,   # Daboia russelii
        "label": "russells_viper",
        "telugu": "అన్నేపాము",
        "target": 1500,
    },
    "common_krait": {
        "taxon_id": 96224,   # Bungarus caeruleus
        "label": "common_krait",
        "telugu": "కట్లపాము",
        "target": 1500,
    },
    "sawscaled_viper": {
        "taxon_id": 31079,   # Echis carinatus
        "label": "sawscaled_viper",
        "telugu": "రాలుపాము",
        "target": 1500,
    },
}

# Non-venomous: common Telangana/India species — taxon IDs verified against iNaturalist API
# Rat Snake (Ptyas mucosa), Checkered Keelback (Fowlea piscator),
# Trinket Snake (Coelognathus helena), Lycodon fasciolatus, Vine Snake (Ahaetulla nasuta),
# Indian Sand Boa (Eryx conicus), Rock Python (Python molurus)
NON_VENOMOUS_TAXON_IDS = [28868, 966799, 30264, 1367260, 68528, 32205, 238252]

NON_VENOMOUS_CLASS = {
    "label": "non_venomous",
    "taxon_ids": NON_VENOMOUS_TAXON_IDS,
    "target": 3000,
}

# India-wide bounding box. PRD's narrow Telangana bbox (12-20N, 76-82E) yields near-zero
# results for rare species; widening to all of India still keeps fauna relevant.
INDIA_BBOX = {
    "nelat": 35.0,
    "nelng": 97.0,
    "swlat": 6.0,
    "swlng": 68.0,
}

# Only download observations with research-grade quality
QUALITY_GRADE = "research"

MIN_DIM = 224          # pixels — PRD §11.2
SNAKE_FRAME_MIN = 0.0  # iNaturalist doesn't expose this; handled by quality_grade
REQUEST_DELAY = 0.5    # seconds between API pages (be polite to the API)
PAGE_SIZE = 100        # max allowed by iNaturalist API
DOWNLOAD_TIMEOUT = 20  # seconds per image download

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# iNaturalist API helpers
# ---------------------------------------------------------------------------

def fetch_observation_page(taxon_id: int, page: int) -> dict:
    """Fetch one page of observations for a taxon, filtered to India bbox."""
    params = {
        "taxon_id": taxon_id,
        "quality_grade": QUALITY_GRADE,
        "photos": "true",
        "per_page": PAGE_SIZE,
        "page": page,
        "order": "desc",
        "order_by": "votes",
        **INDIA_BBOX,
    }
    resp = requests.get(f"{INATURALIST_API}/observations", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def collect_photo_urls(taxon_id: int, max_count: int) -> list[dict]:
    """
    Walk iNaturalist pages and collect up to max_count photo records.
    Returns list of {"url": str, "obs_id": int}.
    """
    records = []
    page = 1

    with tqdm(total=max_count, desc=f"  taxon {taxon_id} — listing", unit="img", leave=False) as pbar:
        while len(records) < max_count:
            try:
                data = fetch_observation_page(taxon_id, page)
            except requests.RequestException as e:
                print(f"\n    API error page {page}: {e} — stopping early")
                break

            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                for photo in obs.get("photos", []):
                    url = photo.get("url", "")
                    if not url:
                        continue
                    # iNaturalist thumbnail URLs end in /square.jpg — get full size
                    url = url.replace("/square.", "/large.")
                    obs_id = obs.get("id", 0)
                    license_code = photo.get("license_code") or ""
                    # skip rights-reserved images
                    if license_code.lower() in ("", "all rights reserved"):
                        continue
                    records.append({"url": url, "obs_id": obs_id})
                    pbar.update(1)
                    if len(records) >= max_count:
                        break
                if len(records) >= max_count:
                    break

            total_available = data.get("total_results", 0)
            if page * PAGE_SIZE >= total_available:
                break

            page += 1
            time.sleep(REQUEST_DELAY)

    return records


# ---------------------------------------------------------------------------
# Image download & validation
# ---------------------------------------------------------------------------

def download_image(url: str, dest: Path) -> bool:
    """
    Download image to dest. Returns True on success.
    Validates minimum dimension (224×224) per PRD §11.2.
    """
    if dest.exists():
        return True  # already have it

    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)

        # validate image opens and meets min dimension
        with Image.open(dest) as img:
            w, h = img.size
            if w < MIN_DIM or h < MIN_DIM:
                dest.unlink()
                return False

        return True

    except (requests.RequestException, UnidentifiedImageError, OSError):
        if dest.exists():
            dest.unlink()
        return False


def download_class(label: str, records: list[dict], class_dir: Path, resume: bool) -> int:
    """Download images for one class. Returns count of successfully saved images."""
    class_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(class_dir.glob("*.jpg"))) if resume else 0
    downloaded = existing
    skipped = 0

    with tqdm(records, desc=f"  {label}", unit="img") as pbar:
        for rec in pbar:
            obs_id = rec["obs_id"]
            url = rec["url"]
            ext = Path(url.split("?")[0]).suffix or ".jpg"
            fname = f"{obs_id}_{abs(hash(url)) % 100000}{ext}"
            dest = class_dir / fname

            if resume and dest.exists():
                skipped += 1
                pbar.set_postfix(saved=downloaded, skipped=skipped)
                continue

            if download_image(url, dest):
                downloaded += 1
            else:
                skipped += 1

            pbar.set_postfix(saved=downloaded, skipped=skipped)
            time.sleep(0.05)  # gentle throttle

    return downloaded


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(class_counts: dict[str, int], out_dir: Path) -> None:
    manifest = {
        "classes": {
            "0": "non_venomous",
            "1": "cobra",
            "2": "russells_viper",
            "3": "common_krait",
            "4": "sawscaled_viper",
        },
        "counts": class_counts,
        "targets": {
            "non_venomous": NON_VENOMOUS_CLASS["target"],
            **{k: v["target"] for k, v in VENOMOUS_CLASSES.items()},
        },
        "bbox": INDIA_BBOX,
        "quality_grade": QUALITY_GRADE,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest written -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download snake images from iNaturalist")
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Cap images per class (0 = use PRD targets). Use 200-500 for a test run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that are already downloaded.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=list(VENOMOUS_CLASSES.keys()) + ["non_venomous"],
        default=None,
        help="Download only specific classes (default: all).",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    class_counts: dict[str, int] = {}

    # --- Venomous classes ---
    for name, cfg in VENOMOUS_CLASSES.items():
        if args.classes and name not in args.classes:
            continue

        target = args.max_per_class if args.max_per_class > 0 else cfg["target"]
        class_dir = DATA_DIR / name

        # Resume fix: count existing images, skip class entirely if already at target
        existing_count = len(list(class_dir.glob("*.*"))) if class_dir.exists() else 0
        if args.resume and existing_count >= target:
            print(f"\n[{name}] {existing_count} images already on disk — skipping")
            class_counts[name] = existing_count
            continue

        # Only fetch enough new URLs to top up to target
        still_needed = max(0, target - existing_count)
        print(f"\n[{name}] taxon {cfg['taxon_id']} — have {existing_count}, need {still_needed} more")

        records = collect_photo_urls(cfg["taxon_id"], still_needed if args.resume else target)
        print(f"  Found {len(records)} candidate URLs")

        count = download_class(name, records, class_dir, resume=args.resume)
        class_counts[name] = count
        print(f"  Saved {count} images -> {class_dir}")

    # --- Non-venomous class ---
    if not args.classes or "non_venomous" in args.classes:
        nv_target = args.max_per_class if args.max_per_class > 0 else NON_VENOMOUS_CLASS["target"]
        class_dir = DATA_DIR / "non_venomous"

        existing_count = len(list(class_dir.glob("*.*"))) if class_dir.exists() else 0
        if args.resume and existing_count >= nv_target:
            print(f"\n[non_venomous] {existing_count} images already on disk — skipping")
            class_counts["non_venomous"] = existing_count
        else:
            still_needed = max(0, nv_target - existing_count)
            per_taxon = max(1, still_needed // len(NON_VENOMOUS_TAXON_IDS))
            print(f"\n[non_venomous] {len(NON_VENOMOUS_TAXON_IDS)} taxon IDs — have {existing_count}, need {still_needed} more")

            all_records: list[dict] = []
            for tid in NON_VENOMOUS_TAXON_IDS:
                recs = collect_photo_urls(tid, per_taxon)
                all_records.extend(recs)
                print(f"  taxon {tid}: {len(recs)} URLs")

            count = download_class("non_venomous", all_records, class_dir, resume=args.resume)
            class_counts["non_venomous"] = count
            print(f"  Saved {count} images -> {class_dir}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("Dataset summary")
    print("=" * 50)
    total = 0
    for label, count in class_counts.items():
        cfg = VENOMOUS_CLASSES.get(label)
        target = cfg["target"] if cfg else NON_VENOMOUS_CLASS["target"]
        pct = count / target * 100 if target else 0
        status = "OK" if count >= target * 0.9 else "LOW"
        print(f"  {label:<20} {count:>5} / {target}  ({pct:.0f}%)  [{status}]")
        total += count
    print(f"  {'TOTAL':<20} {total:>5}")

    write_manifest(class_counts, DATA_DIR)


if __name__ == "__main__":
    main()
