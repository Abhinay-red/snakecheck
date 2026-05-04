"""
zip_for_colab.py

Zips data/raw/ into a single file ready to upload to Google Drive.
Run this once the full dataset download is complete.

Usage:
    python scripts/zip_for_colab.py
    python scripts/zip_for_colab.py --out my_drive_folder/data.zip
"""

import argparse
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "snakecheck_data.zip"

CLASS_NAMES = ["non_venomous", "cobra", "russells_viper", "common_krait", "sawscaled_viper"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Counting images...")
    total = 0
    for cls in CLASS_NAMES:
        d = DATA_DIR / cls
        n = len([f for f in d.glob("*") if f.suffix.lower() in IMG_EXTS]) if d.exists() else 0
        print(f"  {cls:<25} {n:>5} images")
        total += n
    print(f"  {'TOTAL':<25} {total:>5} images\n")

    print(f"Zipping to {args.out} ...")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls in CLASS_NAMES:
            d = DATA_DIR / cls
            if not d.exists():
                print(f"  WARNING: {d} not found, skipping")
                continue
            files = [f for f in d.glob("*") if f.suffix.lower() in IMG_EXTS]
            for f in files:
                # Inside the zip: class_name/filename.jpg
                zf.write(f, arcname=f"{cls}/{f.name}")

    size_mb = args.out.stat().st_size / 1e6
    print(f"Done. {args.out.name}  ({size_mb:.1f} MB)")
    print(f"\nNext: upload {args.out} to Google Drive at:")
    print("  My Drive/snakecheck/data.zip")
    print("Then open SnakeCheck_Training.ipynb in Colab and run all cells.")


if __name__ == "__main__":
    main()
