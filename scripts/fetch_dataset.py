"""
fetch_dataset.py - download SnakeCheck images from iNaturalist in small, tracked batches.

Every image is logged in manifests/catalog.csv with its taxon, observation,
observer, license, URL, hashes and batch name. Because the catalog is committed
to git, the exact dataset can be rebuilt later with `restore`, and each batch
can be judged on its own (train with and without it, compare on the frozen
validation set).

Commands
    resolve                          look up taxon and place ids, write manifests/taxa_resolved.json
    add --batch b001 --per-class 200 download up to 200 NEW images per class (respects caps)
    add --batch t001 --pool external_test --per-class 100
                                     download the held-out Telangana test pool
    restore                          re-download every catalog image that is missing on disk
    status                           counts per class, pool, group and batch

Examples
    python scripts/fetch_dataset.py resolve
    python scripts/fetch_dataset.py add --batch smoke --per-class 5        # quick check
    python scripts/fetch_dataset.py add --batch b001 --per-class 300
    python scripts/fetch_dataset.py add --batch b002 --per-class 300 --classes non_venomous
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone

from data_common import (
    INatClient, Paths, append_catalog, image_fingerprint, load_config, load_json,
    read_catalog, save_json,
)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def resolve_taxon(client: INatClient, name: str) -> dict:
    data = client.get("taxa", {"q": name, "is_active": "true", "per_page": 30})
    results = data.get("results", [])
    for t in results:
        if t.get("name", "").lower() == name.lower():
            return {"id": t["id"], "name": t["name"], "rank": t.get("rank"),
                    "observations_count": t.get("observations_count")}
    # Accept a renamed taxon if iNaturalist says our name is a synonym of it.
    for t in results:
        if (t.get("matched_term") or "").lower() == name.lower():
            print(f"  note: '{name}' is now '{t['name']}' on iNaturalist; using taxon {t['id']}")
            return {"id": t["id"], "name": t["name"], "rank": t.get("rank"),
                    "observations_count": t.get("observations_count"), "config_name": name}
    cands = ", ".join(f"{t.get('name')} ({t.get('id')})" for t in results[:5]) or "none"
    raise SystemExit(f"Could not resolve taxon '{name}'. Candidates: {cands}. Fix config/dataset.yaml.")


def resolve_place(client: INatClient, place) -> dict:
    """Resolve a country or state name (or a numeric id) to an iNaturalist place."""
    if isinstance(place, int):
        return {"id": place, "name": str(place)}
    data = client.get("places/autocomplete", {"q": place, "per_page": 30})
    results = data.get("results", [])
    matches = [p for p in results
               if place.lower() in {(p.get("name") or "").lower(), (p.get("display_name") or "").lower(),
                                    (p.get("display_name") or "").split(",")[0].strip().lower()}
               and p.get("admin_level") in (0, 1, 10)]   # country or state
    if matches:
        p = min(matches, key=lambda p: p.get("admin_level"))
        return {"id": p["id"], "name": p.get("display_name") or p.get("name")}
    cands = ", ".join(f"{p.get('display_name')} ({p.get('id')}, level {p.get('admin_level')})"
                      for p in results[:5]) or "none"
    raise SystemExit(f"Could not resolve place '{place}'. Candidates: {cands}. "
                     "Put the numeric place id in config/dataset.yaml instead.")


def cmd_resolve(cfg: dict, paths: Paths, client: INatClient) -> dict:
    out = {"taxa": {}, "places": {}, "exclude_places": {}}
    for cls, ccfg in cfg["classes"].items():
        for t in ccfg["taxa"]:
            r = resolve_taxon(client, t["name"])
            out["taxa"][t["name"]] = r
            print(f"  {cls:<16} {t['name']:<24} -> {r['id']:>8}  ({r.get('observations_count')} obs)")
    for pool, pcfg in cfg["pools"].items():
        out["places"][pool] = [resolve_place(client, p) for p in pcfg["places"]]
        out["exclude_places"][pool] = [resolve_place(client, p) for p in pcfg.get("exclude_places", [])]
        msg = ", ".join(f"{p['name']} ({p['id']})" for p in out["places"][pool])
        if out["exclude_places"][pool]:
            msg += " minus " + ", ".join(f"{p['name']} ({p['id']})" for p in out["exclude_places"][pool])
        print(f"  pool {pool}: {msg}")
    save_json(paths.resolved, out)
    print(f"Wrote {paths.resolved}")
    return out


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

def split_quota(n: int, taxa: list[dict]) -> list[int]:
    """Split n across taxa by weight (largest remainder)."""
    weights = [float(t.get("weight", 1)) for t in taxa]
    total = sum(weights)
    raw = [n * w / total for w in weights]
    q = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - q[i], reverse=True)
    for i in order[: n - sum(q)]:
        q[i] += 1
    return q


def photo_url(url: str, size: str) -> str:
    for s in ("square", "small", "thumb", "medium", "large", "original"):
        url = url.replace(f"/{s}.", f"/{size}.")
    return url


class Fetcher:
    def __init__(self, cfg: dict, paths: Paths, client: INatClient, batch: str, pool: str):
        self.cfg, self.paths, self.client, self.batch, self.pool = cfg, paths, client, batch, pool
        self.dl = cfg["download"]
        self.limits = cfg["limits"]
        self.resolved = load_json(paths.resolved, None)
        if not self.resolved:
            raise SystemExit("Run `python scripts/fetch_dataset.py resolve` first.")
        if pool not in self.resolved["places"]:
            raise SystemExit(f"Pool '{pool}' is not resolved; run resolve again.")
        self.state = load_json(paths.fetch_state, {})
        catalog = read_catalog(paths.catalog)
        self.seen_photos = {r["image_id"] for r in catalog}
        self.seen_obs = {r["observation_id"] for r in catalog if r["observation_id"]}
        self.seen_sha = {r["sha256"] for r in catalog}
        self.class_count = Counter((r["pool"], r["class"]) for r in catalog)
        self.observer_count = Counter((r["class"], r["observer_id"]) for r in catalog)
        self.licenses = {l.lower() for l in self.dl["licenses"]}
        self.place_ids = ",".join(str(p["id"]) for p in self.resolved["places"][pool])
        self.not_in_place = ",".join(str(p["id"]) for p in self.resolved.get("exclude_places", {}).get(pool, []))
        self.skips = Counter()

    def _take_observation(self, obs: dict, cls: str, group: str, taxon: dict, want: int) -> list[dict]:
        obs_id = str(obs["id"])
        if obs_id in self.seen_obs:
            self.skips["observation already in catalog"] += 1
            return []
        user = obs.get("user") or {}
        observer = str(user.get("id", ""))
        if not observer:
            self.skips["no observer id"] += 1
            return []
        per_observer = self.limits["photos_per_observer_per_class"]
        rows = []
        for photo in obs.get("photos", []):
            if len(rows) >= min(want, self.limits["photos_per_observation"]):
                break
            if self.observer_count[(cls, observer)] >= per_observer:
                self.skips["observer cap reached"] += 1
                break
            lic = (photo.get("license_code") or "").lower()
            if lic not in self.licenses:
                self.skips["license not allowed"] += 1
                continue
            image_id = f"inat:{photo['id']}"
            if image_id in self.seen_photos:
                continue
            url = photo_url(photo.get("url") or "", self.dl["photo_size"])
            data = self.client.download(url)
            if not data:
                self.skips["download failed"] += 1
                continue
            try:
                sha, ph, w, h = image_fingerprint(data)
            except Exception:
                self.skips["unreadable image"] += 1
                continue
            if min(w, h) < self.dl["min_side"]:
                self.skips["too small"] += 1
                continue
            if sha in self.seen_sha:
                self.skips["exact duplicate"] += 1
                continue
            rel = f"images/{cls}/{photo['id']}.jpg"
            dest = self.paths.images.parent / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            coords = (obs.get("geojson") or {}).get("coordinates") or ["", ""]
            rows.append({
                "image_id": image_id, "source": "inaturalist", "pool": self.pool,
                "class": cls, "group": group, "taxon_id": (obs.get("taxon") or {}).get("id", taxon["id"]),
                "taxon_name": (obs.get("taxon") or {}).get("name", taxon["name"]),
                "observation_id": obs_id, "observer_id": observer,
                "observer_login": user.get("login", ""), "license": lic, "url": url,
                "observation_url": obs.get("uri") or f"https://www.inaturalist.org/observations/{obs_id}",
                "observed_on": obs.get("observed_on") or "", "lat": coords[1], "lng": coords[0],
                "width": w, "height": h, "sha256": sha, "phash": ph, "batch": self.batch,
                "path": rel, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            self.seen_photos.add(image_id)
            self.seen_sha.add(sha)
            self.observer_count[(cls, observer)] += 1
        if rows:
            self.seen_obs.add(obs_id)
        return rows

    def fetch_taxon(self, cls: str, group: str, taxon_name: str, want: int) -> int:
        """Download up to `want` new images of one taxon. Returns how many were saved."""
        if want <= 0:
            return 0
        taxon = self.resolved["taxa"].get(taxon_name)
        if taxon is None:
            raise SystemExit(f"'{taxon_name}' is not in taxa_resolved.json; run resolve again.")
        key = f"{self.pool}|{cls}|{taxon_name}"
        st = self.state.setdefault(key, {"id_above": 0, "exhausted": False})
        got = 0
        while got < want and not st["exhausted"]:
            params = {
                "taxon_id": taxon["id"], "place_id": self.place_ids,
                "quality_grade": self.dl["quality_grade"], "photos": "true",
                "captive": str(self.dl.get("captive", False)).lower(),
                "photo_license": ",".join(sorted(self.licenses)),
                "order_by": "id", "order": "asc", "id_above": st["id_above"],
                "per_page": self.dl["page_size"],
            }
            if self.not_in_place:
                params["not_in_place"] = self.not_in_place
            results = self.client.get("observations", params).get("results", [])
            if not results:
                st["exhausted"] = True
                break
            rows: list[dict] = []
            for obs in results:
                new = self._take_observation(obs, cls, group, taxon, want - got)
                rows += new
                got += len(new)
                st["id_above"] = obs["id"]
                if got >= want:
                    break
            # Write after every page so an interrupted run loses nothing.
            append_catalog(self.paths.catalog, rows)
            save_json(self.paths.fetch_state, self.state)
        save_json(self.paths.fetch_state, self.state)
        return got

    def fetch_class(self, cls: str, per_class: int) -> dict:
        ccfg = self.cfg["classes"][cls]
        have = self.class_count[(self.pool, cls)]
        n = per_class
        if self.pool == "main":
            n = max(0, min(per_class, ccfg["cap"] - have))
        if n == 0:
            print(f"[{cls}] at cap ({have}/{ccfg['cap']}); nothing to add")
            return {}
        taxa = ccfg["taxa"]
        quotas = split_quota(n, taxa)
        result = {}
        print(f"[{cls}] adding up to {n} (have {have} in pool '{self.pool}')")
        for t, q in zip(taxa, quotas):
            result[t["name"]] = self.fetch_taxon(cls, t.get("group", "core"), t["name"], q)
        # Second pass: give what rare taxa could not fill to the others.
        short = n - sum(result.values())
        if short > 0:
            for t in sorted(taxa, key=lambda t: -float(t.get("weight", 1))):
                if short <= 0:
                    break
                extra = self.fetch_taxon(cls, t.get("group", "core"), t["name"], short)
                result[t["name"]] += extra
                short -= extra
        for name, c in result.items():
            print(f"    {name:<24} +{c}")
        self.class_count[(self.pool, cls)] += sum(result.values())
        return result


def cmd_add(cfg, paths, client, batch, per_class, pool, classes):
    f = Fetcher(cfg, paths, client, batch, pool)
    total = 0
    for cls in classes or list(cfg["classes"]):
        total += sum(f.fetch_class(cls, per_class).values())
    print(f"\nBatch '{batch}': {total} new images in pool '{pool}'")
    if f.skips:
        print("Skipped: " + ", ".join(f"{k} {v}" for k, v in f.skips.most_common()))
    return total


# ---------------------------------------------------------------------------
# restore / status
# ---------------------------------------------------------------------------

def cmd_restore(paths: Paths, client: INatClient):
    rows = read_catalog(paths.catalog)
    missing = [r for r in rows if not (paths.images.parent / r["path"]).exists()]
    print(f"{len(rows)} images in catalog, {len(missing)} missing on disk")
    bad = 0
    for i, r in enumerate(missing, 1):
        if r["source"] != "inaturalist":
            print(f"  cannot restore {r['image_id']} ({r['source']}); re-add it with add_local_images.py")
            continue
        data = client.download(r["url"])
        dest = paths.images.parent / r["path"]
        if not data:
            bad += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        sha, ph, _, _ = image_fingerprint(data)
        if sha != r["sha256"]:
            print(f"  warning: {r['image_id']} changed since it was catalogued (phash {r['phash']} -> {ph})")
        if i % 100 == 0:
            print(f"  {i}/{len(missing)}")
    print(f"Restored {len(missing) - bad}, failed {bad}")


def cmd_status(paths: Paths):
    rows = read_catalog(paths.catalog)
    if not rows:
        print("Catalog is empty.")
        return
    by = Counter((r["pool"], r["class"], r["group"]) for r in rows)
    print(f"{'pool':<14}{'class':<18}{'group':<15}{'images':>7}")
    for (p, c, g), n in sorted(by.items()):
        print(f"{p:<14}{c:<18}{g:<15}{n:>7}")
    print("\nper batch: " + ", ".join(f"{b} {n}" for b, n in sorted(Counter(r["batch"] for r in rows).items())))
    print(f"observers: {len({r['observer_id'] for r in rows})}, images: {len(rows)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("resolve")
    a = sub.add_parser("add")
    a.add_argument("--batch", required=True, help="name for this increment, e.g. b001")
    a.add_argument("--per-class", type=int, default=100)
    a.add_argument("--pool", default="main")
    a.add_argument("--classes", nargs="+")
    sub.add_parser("restore")
    sub.add_parser("status")
    args = ap.parse_args(argv)

    paths = Paths()
    cfg = load_config(paths.config)
    client = INatClient(api=cfg["api"], delay_s=cfg["download"]["request_delay_s"])
    if args.cmd == "resolve":
        cmd_resolve(cfg, paths, client)
    elif args.cmd == "add":
        bad = [c for c in (args.classes or []) if c not in cfg["classes"]]
        if bad:
            raise SystemExit(f"Unknown class(es): {bad}. Known: {list(cfg['classes'])}")
        cmd_add(cfg, paths, client, args.batch, args.per_class, args.pool, args.classes)
    elif args.cmd == "restore":
        cmd_restore(paths, client)
    else:
        cmd_status(paths)


if __name__ == "__main__":
    main()
