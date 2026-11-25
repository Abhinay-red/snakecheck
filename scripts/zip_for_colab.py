"""
zip_for_colab.py - zip the materialized splits for Google Drive / Colab.

Run build_splits.py --materialize data/splits first. The zip holds
<split>/<class>/<image> plus the manifests, so the notebook trains on train,
tunes on val, and only touches test and test_external once at the end.

    python scripts/zip_for_colab.py
    python scripts/zip_for_colab.py --splits data/splits --out data/snakecheck_data.zip
"""

import argparse
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "snakecheck_data.zip")
    a = ap.parse_args()
    if not a.splits.is_dir():
        raise SystemExit(f"{a.splits} not found. Run: python scripts/build_splits.py --materialize {a.splits}")
    files = [f for f in a.splits.rglob("*") if f.suffix.lower() in IMG_EXTS]
    counts = Counter(f.relative_to(a.splits).parts[:2] for f in files)
    for (split, cls), n in sorted(counts.items()):
        print(f"  {split:<15}{cls:<18}{n:>6}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as zf:   # JPEGs do not compress further
        for f in files:
            zf.write(f, arcname=str(f.relative_to(a.splits)))
        for m in ("eval_frozen.csv", "splits.csv", "split_report.md"):
            p = ROOT / "manifests" / m
            if p.exists():
                zf.write(p, arcname=f"manifests/{m}")
    print(f"Wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB, {len(files)} images)")
    print("Upload it to My Drive/snakecheck/data.zip for the Colab notebook.")


if __name__ == "__main__":
    main()
