"""Offline tests for the data pipeline. A fake iNaturalist serves synthetic photos.

    python -m pytest tests -q
"""

import copy
import io
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_splits  # noqa: E402
import fetch_dataset  # noqa: E402
from add_local_images import add_local  # noqa: E402
from data_common import Paths, load_config, read_catalog  # noqa: E402


def make_image(seed: int, quality=90, size=320) -> bytes:
    rng = np.random.default_rng(seed)
    small = (rng.random((8, 8, 3)) * 255).astype("uint8")
    img = Image.fromarray(small).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class FakeINat:
    """Observations for every (taxon, place) with a mix of busy and occasional observers."""

    def __init__(self, dup_map=None):
        self.dup_map = dup_map or {}   # photo id -> seed of another photo (a repost)
        self.obs = {}
        self.calls = []

    def _observations(self, taxon_id, place_ids):
        key = (taxon_id, place_ids)
        if key not in self.obs:
            rng = random.Random(hash(key) & 0xFFFF)
            base = (taxon_id % 1000) * 100000 + (0 if place_ids == "1" else 50000)
            out = []
            for k in range(300):
                oid = base + k + 1
                # observer 7 is a power user who appears in every class and pool
                user = 7 if k % 10 == 0 else rng.randint(100, 400) + (0 if place_ids == "1" else 1000)
                photos = [{"id": oid * 10 + p, "license_code": "cc-by" if k % 13 else None,
                           "url": f"https://fake/photos/{oid * 10 + p}/square.jpg"} for p in range(3)]
                out.append({"id": oid, "user": {"id": user, "login": f"u{user}"}, "photos": photos,
                            "taxon": {"id": taxon_id, "name": f"t{taxon_id}"}, "observed_on": "2024-01-01",
                            "geojson": {"coordinates": [78.0, 17.0]}})
            self.obs[key] = out
        return self.obs[key]

    def get(self, path, params):
        if path == "taxa":
            return {"results": [{"id": abs(hash(params["q"])) % 99991 + 1, "name": params["q"]}]}
        if path == "places/autocomplete":
            pid = 1 if params["q"] == "India" else 2
            return {"results": [{"id": pid, "name": params["q"], "admin_level": 0 if pid == 1 else 1}]}
        self.calls.append(params)
        allp = self._observations(params["taxon_id"], params["place_id"])
        res = [o for o in allp if o["id"] > params["id_above"]]
        return {"results": res[: params["per_page"]]}

    def download(self, url):
        pid = int(url.split("/")[-2])
        if pid in self.dup_map:
            return make_image(self.dup_map[pid], quality=70)
        return make_image(pid)


@pytest.fixture
def env(tmp_path):
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["pools"] = {"main": {"places": ["India"], "exclude_places": ["Telangana"]},
                    "external_test": {"places": ["Telangana"]}}
    cfg["splits"]["min_eval_per_class"] = 3
    paths = Paths(root=tmp_path, config=Path("unused"), manifests=tmp_path / "manifests",
                  images=tmp_path / "data" / "images")
    return cfg, paths


def run_add(cfg, paths, client, batch, n, pool="main", classes=None):
    return fetch_dataset.cmd_add(cfg, paths, client, batch, n, pool, classes)


def test_download_logs_provenance_and_respects_limits(env):
    cfg, paths = env
    client = FakeINat()
    fetch_dataset.cmd_resolve(cfg, paths, client)
    run_add(cfg, paths, client, "b001", 40)
    rows = read_catalog(paths.catalog)
    per_class = Counter(r["class"] for r in rows)
    assert all(per_class[c] == 40 for c in cfg["classes"]), per_class
    for r in rows:
        assert r["license"] == "cc-by" and r["observer_id"] and r["observation_id"] and r["url"]
        assert (paths.images.parent / r["path"]).exists()
    assert max(Counter(r["observation_id"] for r in rows).values()) <= cfg["limits"]["photos_per_observation"]
    assert max(Counter((r["class"], r["observer_id"]) for r in rows).values()) \
        <= cfg["limits"]["photos_per_observer_per_class"]
    # main pool asks for India minus Telangana; the test pool asks for Telangana only
    assert all(c["place_id"] == "1" and c["not_in_place"] == "2" for c in client.calls)
    hard = [r for r in rows if r["group"] == "hard_negative"]
    assert len(hard) > 20                                     # hard negatives get most of non_venomous


def test_increments_continue_and_respect_cap(env):
    cfg, paths = env
    cfg["classes"]["cobra"]["cap"] = 50
    client = FakeINat()
    fetch_dataset.cmd_resolve(cfg, paths, client)
    run_add(cfg, paths, client, "b001", 30, classes=["cobra"])
    run_add(cfg, paths, client, "b002", 30, classes=["cobra"])
    rows = read_catalog(paths.catalog)
    assert Counter(r["batch"] for r in rows) == {"b001": 30, "b002": 20}
    assert len({r["image_id"] for r in rows}) == 50


def test_splits_are_leak_free_and_frozen(env):
    cfg, paths = env
    client = FakeINat()
    fetch_dataset.cmd_resolve(cfg, paths, client)
    run_add(cfg, paths, client, "b001", 120)
    run_add(cfg, paths, client, "t001", 30, pool="external_test")

    build_splits.build(cfg, paths, quiet=True)
    assert not paths.eval_frozen.exists()                      # preview never freezes

    rows = build_splits.build(cfg, paths, freeze=True, force_small=True, quiet=True)
    split_of_observer = {}
    for r in rows:
        if r["split"] in ("train", "val", "test", "test_external"):
            assert split_of_observer.setdefault(r["observer_id"], r["split"]) == r["split"]
    s = Counter(r["split"] for r in rows)
    assert s["val"] > 0 and s["test"] > 0 and s["test_external"] > 0 and s["train"] > s["val"]
    # power user 7 is in both pools, so their external photos must be dropped
    assert all(r["split"] == "excluded" for r in rows if r["pool"] == "external_test" and r["observer_id"] == "7")

    frozen_before = {r["image_id"]: r["split"] for r in read_catalog(paths.eval_frozen)}
    with pytest.raises(SystemExit):
        build_splits.build(cfg, paths, freeze=True, quiet=True)

    run_add(cfg, paths, client, "b002", 120)
    rows = build_splits.build(cfg, paths, quiet=True)
    after = {r["image_id"]: r["split"] for r in rows if r["reason"] == "frozen"}
    assert after == frozen_before                              # new data never changes val/test
    eval_observers = {r["observer_id"] for r in rows if r["split"] in ("val", "test")}
    new = [r for r in rows if r["batch"] == "b002"]
    assert new and all(r["split"] in ("train", "excluded") for r in new)
    assert all(r["split"] == "excluded" for r in new if r["observer_id"] in eval_observers)

    rows = build_splits.build(cfg, paths, train_batches=["b001"], quiet=True)
    assert {r["batch"] for r in rows if r["split"] == "train"} == {"b001"}

    out = paths.root / "splits"
    build_splits.materialize(rows, paths, out)
    assert (out / "val").is_dir() and (out / "train" / "cobra").is_dir()


def test_near_duplicates_are_removed(env):
    cfg, paths = env
    # A repost: photo 2 of one observation is a recompressed copy of another class's photo.
    client = FakeINat()
    fetch_dataset.cmd_resolve(cfg, paths, client)
    run_add(cfg, paths, client, "b001", 10, classes=["cobra", "common_krait"])
    rows = read_catalog(paths.catalog)
    cobra = next(r for r in rows if r["class"] == "cobra")
    krait = next(r for r in rows if r["class"] == "common_krait")
    same = [r for r in rows if r["class"] == "cobra" and r["image_id"] != cobra["image_id"]][0]
    # Simulate reposts by copying hashes: a same-class duplicate and a cross-class duplicate.
    same["phash"] = cobra["phash"]
    other = [r for r in rows if r["class"] == "cobra" and r["image_id"] not in (cobra["image_id"], same["image_id"])][0]
    krait["phash"] = other["phash"]
    from data_common import CATALOG_FIELDS, write_csv
    write_csv(paths.catalog, rows, CATALOG_FIELDS)
    out = {r["image_id"]: r for r in build_splits.build(cfg, paths, quiet=True)}
    assert out[same["image_id"]]["split"] == "excluded"
    assert "near duplicate" in out[same["image_id"]]["reason"]
    assert out[cobra["image_id"]]["group_id"] == out[same["image_id"]]["group_id"]
    assert out[cobra["image_id"]]["split"] != "excluded"
    for r in (other, krait):
        assert out[r["image_id"]]["reason"] == "duplicate photo with conflicting labels"


def test_real_phash_catches_recompressed_copy():
    from data_common import image_fingerprint
    _, a, _, _ = image_fingerprint(make_image(5))
    _, b, _, _ = image_fingerprint(make_image(5, quality=60, size=500))
    _, c, _, _ = image_fingerprint(make_image(6))
    pairs = build_splits.near_duplicate_pairs([a, b, c], 6)
    assert pairs == [(0, 1)]


def test_local_images_go_to_external_test(env, tmp_path):
    cfg, paths = env
    folder = tmp_path / "mine"
    folder.mkdir()
    for k in range(5):
        (folder / f"{k}.jpg").write_bytes(make_image(10_000 + k))
    assert add_local(cfg, paths, folder, "not_a_snake", "me", "own_photos", "own", "local001",
                     "external_test") == 5
    assert add_local(cfg, paths, folder, "not_a_snake", "me", "own_photos", "own", "local001",
                     "external_test") == 0           # re-adding the same files does nothing
