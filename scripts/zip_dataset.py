"""
zip_dataset.py - zip the materialized splits for Kaggle (or Google Colab).

Run build_splits.py --materialize data/splits first. The zip holds

    train/ val/ test/ test_external/   <class>/<image>
    manifests/                         eval_frozen.csv, splits.csv, split_report.md
    code/                              train.py, snake_eval.py, config/dataset.yaml

The training code is inside the zip so the notebook works on Kaggle with
internet turned off, and so every dataset version carries the exact code and
class list it was made for.

    python scripts/zip_dataset.py
    python scripts/zip_dataset.py --splits data/splits --out data/snakecheck_data.zip
"""

import argparse
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CODE = ["scripts/train.py", "scripts/snake_eval.py", "config/dataset.yaml"]


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
            zf.write(f, arcname=f.relative_to(a.splits).as_posix())
        for m in ("eval_frozen.csv", "splits.csv", "split_report.md"):
            p = ROOT / "manifests" / m
            if p.exists():
                zf.write(p, arcname=f"manifests/{m}")
        for c in CODE:
            zf.write(ROOT / c, arcname=f"code/{Path(c).name}")
    print(f"Wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB, {len(files)} images)")
    print("Kaggle: upload it as a new dataset (or a new version of it). Colab: My Drive/snakecheck/data.zip.")


if __name__ == "__main__":
    main()
