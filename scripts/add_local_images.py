"""
add_local_images.py - add photos that do not come from iNaturalist.

Use it for your own photos, SnakeCLEF images, or "not a snake" photos such as
ropes, hoses, sticks and lizards. By default they go to the external test pool
(test_external), which is the honest check of how the model does on photos
from a different source. Use --pool main only if you want them in training.

Each folder you add should hold photos of one class taken by one person or
source; that name is used as the observer so all of it lands in one split.

    python scripts/add_local_images.py --dir ~/Pictures/rat_snakes --class non_venomous \\
        --observer me --source own_photos --batch local001
    python scripts/add_local_images.py --dir snakeclef/naja_naja --class cobra \\
        --observer snakeclef --source snakeclef --license cc-by-nc --batch clef001
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from data_common import Paths, append_catalog, image_fingerprint, load_config, read_catalog

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def add_local(cfg, paths: Paths, folder: Path, cls: str, observer: str, source: str,
              license_: str, batch: str, pool: str) -> int:
    if cls not in cfg["classes"]:
        raise SystemExit(f"Unknown class {cls}. Known: {list(cfg['classes'])}")
    seen = {r["sha256"] for r in read_catalog(paths.catalog)}
    rows = []
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        data = f.read_bytes()
        try:
            sha, ph, w, h = image_fingerprint(data)
        except Exception:
            print(f"  skip unreadable {f}")
            continue
        if sha in seen or min(w, h) < cfg["download"]["min_side"]:
            continue
        rel = f"images/{cls}/local_{sha[:16]}{f.suffix.lower()}"
        dest = paths.images.parent / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        seen.add(sha)
        rows.append({
            "image_id": f"local:{sha[:16]}", "source": source, "pool": pool, "class": cls,
            "group": "core", "taxon_name": "", "observation_id": f"local:{sha[:16]}",
            "observer_id": f"{source}:{observer}", "observer_login": observer, "license": license_,
            "url": "", "observation_url": "", "width": w, "height": h, "sha256": sha, "phash": ph,
            "batch": batch, "path": rel, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    append_catalog(paths.catalog, rows)
    print(f"Added {len(rows)} images to pool '{pool}' as class '{cls}'")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--observer", required=True, help="who took the photos, e.g. me, a friend's name, snakeclef")
    ap.add_argument("--source", default="own_photos")
    ap.add_argument("--license", default="own", help="license of these photos")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--pool", default="external_test", choices=["external_test", "main"])
    a = ap.parse_args(argv)
    paths = Paths()
    add_local(load_config(paths.config), paths, a.dir.expanduser(), a.cls, a.observer, a.source,
              a.license, a.batch, a.pool)


if __name__ == "__main__":
    main()
