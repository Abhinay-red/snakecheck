# SnakeCheck

SnakeCheck tells you from a photo whether a snake is venomous or not. The plan is a model that runs offline on a phone, then a mobile app built around it.

## Goal (from the PRD)

- Classify snake photos as venomous vs non-venomous (the PRD describes a binary classifier).
- Model: EfficientNetB0 transfer learning on iNaturalist photos.
- Deploy on-device as a TFLite model inside an Expo / React Native app. An optional FastAPI backend is allowed.
- Targets: venomous recall at least 0.90, ROC-AUC at least 0.92.

## What is in this repo

The code was copied from an earlier version of the project (github.com/Abhinay-red/snakecheck). That version uses 5 classes for India instead of the PRD's binary setup: `non_venomous`, `cobra`, `russells_viper`, `common_krait`, `sawscaled_viper`.

| File | What it does |
|---|---|
| `scripts/dataset_download.py` | Downloads research-grade iNaturalist photos inside an India bounding box, license-filtered, min 224 px, into `data/raw/<class>/`. Supports `--max_per_class` and `--resume`. |
| `scripts/zip_for_colab.py` | Zips `data/raw/` into one file to upload to Google Drive for Colab. |
| `scripts/train.py` | Local version of the two-phase EfficientNetB0 training (head only, then top 40 layers unfrozen). |
| `SnakeCheck_Training.ipynb` | Colab notebook: load data from Drive, train in two phases, evaluate against launch gates, export INT8 TFLite. Saved outputs from the last run are kept. |

## Status

The original dataset (about 11,800 images) is lost, so data has to be downloaded again with `dataset_download.py`.

Results from the last run, from the notebook outputs:

- Macro F1 0.917, ROC-AUC 0.991, accuracy 0.91.
- All four venomous classes passed their 0.90 recall gate.
- Non-venomous recall was 0.82 raw and 0.854 after threshold tuning, against a 0.92 gate. The gate check failed, so the TFLite export was never run.

## To do

- [ ] Re-download the dataset from iNaturalist.
- [ ] Evaluate on a real held-out test split. The old notebook evaluated on the full dataset (and used it as validation data during training), so the numbers above are optimistic.
- [ ] Fix the threshold tuning direction. The 0.52 venomous threshold pushes predictions away from venomous, which is the wrong bias for a safety app.
- [ ] Add more, and more diverse, non-venomous data. The main error is rat snakes being classified as cobras.
- [ ] Decide between the PRD's binary classifier and the 5-class setup.
- [ ] Add `requirements.txt` for local runs.
- [ ] Export TFLite once the gates pass, then build the mobile app.
