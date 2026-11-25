"""
build_splits.py - leak-free train / val / test splits with a frozen evaluation set.

How the split is made
  1. Near-duplicate photos are found with a perceptual hash. Copies of the same
     photo with the same label keep one image; copies with different labels are
     all dropped.
  2. Images are grouped: every photo by the same observer, and every photo that
     is a near-duplicate of another, ends up in the same group. A group goes to
     exactly one split, so no person's photos (or reposts) leak across splits.
  3. Groups from the `main` pool (India minus Telangana) are sampled into val and test.
     The `external_test` pool (Telangana, your own photos) becomes
     `test_external`. External groups that share an observer or photo with the
     main pool are dropped.
  4. `--freeze` writes val, test and test_external to manifests/eval_frozen.csv
     once. After that, new data only ever goes to train. New photos from an
     observer who is already in val or test are kept out of training.

Usage
    python scripts/build_splits.py                     # preview, writes the report only
    python scripts/build_splits.py --freeze            # first time: freeze val/test
    python scripts/build_splits.py                     # later: refresh train after new batches
    python scripts/build_splits.py --train-batches b001 b002   # train on chosen batches only
    python scripts/build_splits.py --materialize data/splits   # folders for Keras / Colab
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_common import Paths, load_config, read_catalog, write_csv

EVAL_SPLITS = ("val", "test", "test_external")
SPLIT_FIELDS = ["image_id", "class", "split", "reason", "group_id", "observer_id",
                "observation_id", "pool", "group", "batch", "license", "path"]
FROZEN_FIELDS = ["image_id", "class", "split", "group_id", "observer_id", "observation_id",
                 "pool", "source", "group", "taxon_name", "license", "sha256", "phash", "frozen_at"]
MARKER = ".snakecheck_materialized"


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1)


def near_duplicate_pairs(phashes: list[str], max_dist: int, chunk: int = 2048) -> list[tuple[int, int]]:
    h = np.array([int(p, 16) for p in phashes], dtype=np.uint64)
    pairs = []
    for s in range(0, len(h), chunk):
        block = h[s:s + chunk]
        d = _popcount(block[:, None] ^ h[None, :])
        ii, jj = np.nonzero(d <= max_dist)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j)))
    return pairs


def build(cfg: dict, paths: Paths, freeze=False, refreeze=False, force_small=False,
          train_batches=None, extend_external=False, quiet=False):
    scfg = cfg["splits"]
    rows = read_catalog(paths.catalog)
    if not rows:
        raise SystemExit("Catalog is empty; download some data first.")
    n = len(rows)
    for r in rows:
        r["split"], r["reason"] = "", ""

    frozen_rows = read_catalog(paths.eval_frozen) if paths.eval_frozen.exists() else []
    if frozen_rows and freeze and not refreeze:
        raise SystemExit(f"{paths.eval_frozen} already exists. Val/test are frozen; "
                         "use --refreeze only if you really mean to throw them away.")
    if refreeze:
        frozen_rows = []
    frozen = {r["image_id"]: r["split"] for r in frozen_rows}
    is_frozen = bool(frozen)

    # 1. Near duplicates (frozen eval images are never the ones dropped) -----
    pairs = near_duplicate_pairs([r["phash"] for r in rows], scfg["near_duplicate_hamming"])
    dup = DSU(n)
    for i, j in pairs:
        dup.union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[dup.find(i)].append(i)
    for members in clusters.values():
        if len(members) < 2:
            continue
        if len({rows[i]["class"] for i in members}) > 1:
            for i in members:
                if rows[i]["image_id"] not in frozen:
                    rows[i]["split"], rows[i]["reason"] = "excluded", "duplicate photo with conflicting labels"
            continue
        members.sort(key=lambda i: (rows[i]["image_id"] not in frozen, rows[i]["pool"] != "main",
                                    rows[i]["fetched_at"], rows[i]["image_id"]))
        for i in members[1:]:
            if rows[i]["image_id"] not in frozen:
                rows[i]["split"] = "excluded"
                rows[i]["reason"] = f"near duplicate of {rows[members[0]]['image_id']}"

    # 2. Groups: same observer or duplicate photo -> same group ---------------
    grp = DSU(n)
    first_by_observer = {}
    for i, r in enumerate(rows):
        o = r["observer_id"]
        if o in first_by_observer:
            grp.union(i, first_by_observer[o])
        else:
            first_by_observer[o] = i
    for i, j in pairs:
        grp.union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[grp.find(i)].append(i)
    for members in groups.values():
        gid = "g" + min(rows[i]["image_id"] for i in members).replace(":", "_")
        for i in members:
            rows[i]["group_id"] = gid

    # 3. Frozen eval set, or a new assignment ----------------------------------
    if is_frozen:
        missing = set(frozen) - {r["image_id"] for r in rows}
        if missing:
            raise SystemExit(f"{len(missing)} frozen eval images are not in the catalog; "
                             "the catalog must stay append-only.")
        for members in groups.values():
            fz = {frozen[rows[i]["image_id"]] for i in members if rows[i]["image_id"] in frozen}
            has_main = any(rows[i]["pool"] == "main" for i in members)
            for i in members:
                r = rows[i]
                if r["image_id"] in frozen:
                    r["split"], r["reason"] = frozen[r["image_id"]], "frozen"
                    continue
                if r["split"] == "excluded":
                    continue
                if fz == {"test_external"} and extend_external and r["pool"] != "main":
                    r["split"], r["reason"] = "test_external", "added to external test"
                elif fz:
                    r["split"], r["reason"] = "excluded", f"same observer/photo as frozen {'/'.join(sorted(fz))}"
                elif r["pool"] == "main":
                    r["split"], r["reason"] = "train", "not in frozen eval set"
                elif has_main:
                    # Same observer/photo as training data: never put it in the external test.
                    r["split"], r["reason"] = "excluded", "external photo shares observer/photo with main pool"
                elif extend_external:
                    r["split"], r["reason"] = "test_external", "added to external test"
                else:
                    r["split"], r["reason"] = "excluded", "new external data (use --extend-external)"
    else:
        assign_new(rows, groups, scfg)

    # 4. Optional: train on selected batches only -----------------------------
    if train_batches:
        keep = set(train_batches)
        for r in rows:
            if r["split"] == "train" and r["batch"] not in keep:
                r["split"], r["reason"] = "excluded", "batch not selected for training"

    check_no_leak(rows)
    counts = Counter((r["class"], r["split"]) for r in rows)
    classes = list(cfg["classes"])

    if freeze or refreeze:
        small = [(c, s, counts[(c, s)]) for c in classes for s in ("val", "test")
                 if counts[(c, s)] < scfg["min_eval_per_class"]]
        if small and not force_small:
            msg = ", ".join(f"{c}/{s}={k}" for c, s, k in small)
            raise SystemExit(f"Refusing to freeze: too few eval images ({msg}; minimum "
                             f"{scfg['min_eval_per_class']}). Download more first, or pass --force-small.")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        frows = [dict(r, frozen_at=now) for r in rows if r["split"] in EVAL_SPLITS]
        write_csv(paths.eval_frozen, frows, FROZEN_FIELDS)
        is_frozen = True
        if not quiet:
            print(f"Froze {len(frows)} eval images -> {paths.eval_frozen}")

    if is_frozen:
        write_csv(paths.splits, rows, SPLIT_FIELDS)
    report = make_report(rows, classes, is_frozen)
    paths.report.write_text(report, encoding="utf-8")
    if not quiet:
        print(report)
        if not is_frozen:
            print("PREVIEW ONLY: nothing is frozen yet. Run again with --freeze when the counts look right.")
    return rows


def assign_new(rows, groups, scfg):
    """Randomly sample whole observer groups into val and test until each class hits its target."""
    rng = random.Random(scfg["seed"])
    main_total = Counter(r["class"] for r in rows if r["pool"] == "main" and r["split"] != "excluded")
    need = {s: {c: frac * k for c, k in main_total.items()}
            for s, frac in (("val", scfg["val_fraction"]), ("test", scfg["test_fraction"]))}
    keys = sorted(groups, key=lambda g: min(rows[i]["image_id"] for i in groups[g]))
    rng.shuffle(keys)
    for g in keys:
        members = groups[g]
        main = [i for i in members if rows[i]["pool"] == "main"]
        ext = [i for i in members if rows[i]["pool"] != "main"]
        for i in ext:
            if rows[i]["split"] == "excluded":
                continue
            if main:
                rows[i]["split"], rows[i]["reason"] = "excluded", "external photo shares observer/photo with main pool"
            else:
                rows[i]["split"], rows[i]["reason"] = "test_external", "external pool"
        live = [i for i in main if rows[i]["split"] != "excluded"]
        cnt = Counter(rows[i]["class"] for i in live)
        target = "train"
        if cnt:
            for s in rng.sample(["val", "test"], 2):
                if all(k <= need[s][c] + max(1.0, 0.02 * need[s][c]) for c, k in cnt.items()) \
                        and any(need[s][c] > 0 for c in cnt):
                    target = s
                    for c, k in cnt.items():
                        need[s][c] -= k
                    break
        for i in main:
            if rows[i]["split"] != "excluded":
                rows[i]["split"], rows[i]["reason"] = target, "sampled by observer group"


def check_no_leak(rows):
    used = [r for r in rows if r["split"] not in ("excluded", "")]
    for key in ("observer_id", "group_id", "observation_id"):
        where = defaultdict(set)
        for r in used:
            # Two frozen images can only be linked later by a new photo that bridges them;
            # the new photo is kept out of training, so that is reported, not fatal.
            if key == "group_id" and r["reason"] == "frozen":
                continue
            if r[key]:
                where[r[key]].add(r["split"])
        bad = [k for k, s in where.items() if len(s) > 1]
        if bad:
            raise SystemExit(f"LEAK: {len(bad)} {key} values appear in more than one split, e.g. {bad[:3]}")


def make_report(rows, classes, frozen):
    splits = ["train", "val", "test", "test_external", "excluded"]
    c = Counter((r["class"], r["split"]) for r in rows)
    lines = ["# Split report", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             + ("Val/test are FROZEN." if frozen else "PREVIEW: nothing frozen yet."), "",
             "| class | " + " | ".join(splits) + " |", "|---" * (len(splits) + 1) + "|"]
    for cls in classes:
        lines.append(f"| {cls} | " + " | ".join(str(c[(cls, s)]) for s in splits) + " |")
    lines.append("| **total** | " + " | ".join(str(sum(c[(k, s)] for k in classes)) for s in splits) + " |")
    obs = {s: len({r["observer_id"] for r in rows if r["split"] == s}) for s in splits[:4]}
    lines += ["", "Observers per split: " + ", ".join(f"{s} {k}" for s, k in obs.items()),
              "Observer overlap between splits: 0 (checked)", ""]
    hn = Counter(r["split"] for r in rows if r["group"] == "hard_negative")
    lines.append("Hard negatives (look-alike harmless snakes): "
                 + ", ".join(f"{s} {hn[s]}" for s in splits[:4]))
    top = []
    for cls in classes:
        tr = [r for r in rows if r["class"] == cls and r["split"] == "train"]
        if tr:
            o, k = Counter(r["observer_id"] for r in tr).most_common(1)[0]
            top.append(f"{cls} {100 * k / len(tr):.1f}%")
    lines.append("Largest single-observer share of train: " + ", ".join(top))
    reasons = Counter(r["reason"].split(" of ")[0] for r in rows if r["split"] == "excluded")
    if reasons:
        lines += ["", "Excluded: " + "; ".join(f"{k} ({v})" for k, v in reasons.most_common())]
    return "\n".join(lines) + "\n"


def materialize(rows, paths: Paths, out: Path):
    out = out.resolve()
    if out.exists():
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not made by this script; pick an empty folder.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("made by build_splits.py; safe to delete\n")
    missing = 0
    for r in rows:
        if r["split"] in ("excluded", ""):
            continue
        src = paths.images.parent / r["path"]
        if not src.exists():
            missing += 1
            continue
        dst = out / r["split"] / r["class"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    print(f"Materialized splits in {out}" + (f" ({missing} files missing; run fetch_dataset.py restore)" if missing else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true", help="freeze val/test the first time")
    ap.add_argument("--refreeze", action="store_true", help="throw away the frozen eval set and make a new one")
    ap.add_argument("--force-small", action="store_true", help="allow freezing tiny eval sets (testing only)")
    ap.add_argument("--extend-external", action="store_true", help="add new external photos to test_external")
    ap.add_argument("--train-batches", nargs="+", help="only use these batches for training")
    ap.add_argument("--materialize", type=Path, help="write split/class folders here (hard links)")
    args = ap.parse_args(argv)
    paths = Paths()
    cfg = load_config(paths.config)
    rows = build(cfg, paths, args.freeze, args.refreeze, args.force_small, args.train_batches, args.extend_external)
    if args.materialize:
        if not paths.eval_frozen.exists():
            raise SystemExit("Freeze the eval set first (--freeze).")
        materialize(rows, paths, args.materialize)


if __name__ == "__main__":
    main()
