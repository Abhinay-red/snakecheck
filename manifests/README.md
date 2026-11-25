# Manifests

Committed record of the dataset. Do not edit by hand.

- `catalog.csv`: every image ever fetched (append-only), with source, license, observer and hashes.
- `taxa_resolved.json`: iNaturalist ids for the names in `config/dataset.yaml`.
- `fetch_state.json`: where each download stopped, so the next batch continues from there.
- `eval_frozen.csv`: the frozen val, test and test_external images. Never regenerate it casually.
- `splits.csv` and `split_report.md`: the current split, rebuilt by `build_splits.py`.
