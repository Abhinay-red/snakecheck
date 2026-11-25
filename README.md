# SnakeCheck

SnakeCheck tells you from a photo whether a snake is venomous or not. The plan is a model that runs offline on a phone, then a mobile app built around it.

## Goal (from the PRD)

- Classify snake photos as venomous vs non-venomous (the PRD describes a binary classifier).
- Model: EfficientNetB0 transfer learning on iNaturalist photos.
- Deploy on-device as a TFLite model inside an Expo / React Native app. An optional FastAPI backend is allowed.
- Targets: venomous recall at least 0.90, ROC-AUC at least 0.92.

## What is in this repo

| File | What it does |
|---|---|
| `config/dataset.yaml` | The dataset definition: classes, species per class (including look-alike hard negatives), countries per pool, licenses, caps and split settings. |
| `scripts/fetch_dataset.py` | Downloads iNaturalist photos in small named batches and logs every image in `manifests/catalog.csv`. Can restore the whole dataset from the catalog. |
| `scripts/add_local_images.py` | Adds photos from elsewhere (your own photos, SnakeCLEF, ropes and sticks for `not_a_snake`), by default into the external test set. |
| `scripts/build_splits.py` | Removes near-duplicate photos, splits by observer, freezes val/test in `manifests/eval_frozen.csv`, and can write `split/class/` folders. |
| `scripts/zip_for_colab.py` | Zips the split folders and manifests for Google Drive / Colab. |
| `scripts/train.py` | Local version of the two-phase EfficientNetB0 training (not yet updated for the new splits). |
| `SnakeCheck_Training.ipynb` | Colab notebook from the last run (not yet updated for the new splits). |
| `tests/test_data_pipeline.py` | Offline tests of the data pipeline against a fake iNaturalist. |

## Building the dataset

The last dataset had three problems: it was lost, the split leaked (train,
val and test came from the same source, so the scores were too good), and one
big pull of extra data made the model worse with no way to tell which images
did it. The pipeline below is built around those.

**What it guarantees**

- **Nothing is lost.** `manifests/` is committed. It lists every image with its
  URL, observation, observer, license and hashes, so `fetch_dataset.py restore`
  rebuilds the exact same dataset on any machine.
- **No leaks.** All photos by one observer, and all near-duplicate photos
  (perceptual hash), go to the same split. The script stops with an error if
  any observer, observation or duplicate group shows up in two splits.
- **A real test from another source.** `test` is India photos from observers
  the model never saw. `test_external` comes from other countries (Sri Lanka,
  Nepal, Bangladesh, Pakistan) plus any photos you add yourself. Neither is
  used for training or tuning; tune thresholds on `val` only.
- **Frozen evaluation.** After `--freeze`, val, test and test_external never
  change. New batches only go to train, and new photos from an observer who is
  already in val or test are kept out of training.
- **Small, tracked increments.** Each download is a named batch with a per-class
  limit, and classes have caps so they grow together. Train with and without a
  batch (`--train-batches`) and keep it only if val gets better.
- **Balance against the old failure.** About two thirds of `non_venomous` is
  look-alikes (rat snakes, keelbacks, wolf snakes, sand boas, cat snakes,
  kukri snakes), since rat snakes being called cobras was the main error.
  No observer can give more than 25 photos to one class, and at most 2 photos
  are taken per observation.
- **`not_a_snake`.** Lizards, worms, caecilians and eels from iNaturalist, plus
  your own ropes, hoses and sticks, so the app can say "not a snake / unsure".

**How to run it** (on your own computer; iNaturalist is blocked from the cloud
sessions)

```bash
pip install -r requirements-data.txt

# 1. Look up taxon and country ids once (writes manifests/taxa_resolved.json)
python scripts/fetch_dataset.py resolve

# 2. Smoke test: a few images per class, then look at data/images/
python scripts/fetch_dataset.py add --batch smoke --per-class 5
python scripts/fetch_dataset.py status

# 3. First real batch, then the external test pool
python scripts/fetch_dataset.py add --batch b001 --per-class 600
python scripts/fetch_dataset.py add --batch t001 --pool external_test --per-class 150

# 4. Preview the split, then freeze val/test once the counts look right
python scripts/build_splits.py
python scripts/build_splits.py --freeze
git add manifests && git commit -m "Freeze eval set"

# 5. Make folders for training and zip them for Colab
python scripts/build_splits.py --materialize data/splits
python scripts/zip_for_colab.py

# Later: grow in small steps, one batch at a time
python scripts/fetch_dataset.py add --batch b002 --per-class 300
python scripts/build_splits.py --materialize data/splits
git add manifests && git commit -m "Add batch b002"
```

Downloads go at about one request per second, as iNaturalist asks, so a
600-per-class batch takes roughly an hour. You can stop and rerun `add` with
the same batch name; it continues where it stopped.

Your own photos (one folder per class and photographer):

```bash
python scripts/add_local_images.py --dir path/to/ropes --class not_a_snake \
    --observer me --batch local001
python scripts/build_splits.py --extend-external
```

`manifests/split_report.md` shows counts per class and split, observers per
split, hard negatives, the biggest single-observer share, and why images were
excluded. Run the tests with `python -m pytest tests -q`.

## Status

The original dataset (about 11,800 images) is lost. The new data pipeline is in place (see above); the data still has to be downloaded.

Results from the last run, from the notebook outputs:

- Macro F1 0.917, ROC-AUC 0.991, accuracy 0.91.
- All four venomous classes passed their 0.90 recall gate.
- Non-venomous recall was 0.82 raw and 0.854 after threshold tuning, against a 0.92 gate. The gate check failed, so the TFLite export was never run.

## To do

- [x] Leak-free, reproducible data pipeline with a frozen eval set.
- [ ] Download the dataset and freeze the eval set.
- [ ] Update `train.py` and the notebook to read `train/` and `val/` folders instead of splitting by image, and to report on `test` and `test_external`.
- [ ] Evaluate on a real held-out test split. The old notebook evaluated on the full dataset (and used it as validation data during training), so the numbers above are optimistic.
- [ ] Fix the threshold tuning direction. The 0.52 venomous threshold pushes predictions away from venomous, which is the wrong bias for a safety app.
- [ ] Decide between the PRD's binary classifier and the 5-class setup.
- [ ] Add `requirements.txt` for training (`requirements-data.txt` covers the data scripts).
- [ ] Export TFLite once the gates pass, then build the mobile app.
