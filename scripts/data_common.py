"""Shared helpers for the SnakeCheck data scripts: config, catalog, iNaturalist client, hashing."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "dataset.yaml"
MANIFEST_DIR = ROOT / "manifests"
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"

CATALOG_FIELDS = [
    "image_id", "source", "pool", "class", "group", "taxon_id", "taxon_name",
    "observation_id", "observer_id", "observer_login", "license", "url",
    "observation_url", "observed_on", "lat", "lng", "width", "height",
    "sha256", "phash", "batch", "path", "fetched_at",
]


# ---------------------------------------------------------------------------
# Paths (overridable so tests can run in a temp folder)
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    root: Path = ROOT
    config: Path = CONFIG_PATH
    manifests: Path = MANIFEST_DIR
    images: Path = IMAGES_DIR

    @property
    def catalog(self) -> Path:
        return self.manifests / "catalog.csv"

    @property
    def resolved(self) -> Path:
        return self.manifests / "taxa_resolved.json"

    @property
    def fetch_state(self) -> Path:
        return self.manifests / "fetch_state.json"

    @property
    def eval_frozen(self) -> Path:
        return self.manifests / "eval_frozen.csv"

    @property
    def splits(self) -> Path:
        return self.manifests / "splits.csv"

    @property
    def report(self) -> Path:
        return self.manifests / "split_report.md"


def load_config(path: Path = CONFIG_PATH) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Catalog: one row per image ever fetched. Append-only.
# ---------------------------------------------------------------------------

def read_catalog(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_catalog(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Image hashing
# ---------------------------------------------------------------------------

def image_fingerprint(data: bytes) -> tuple[str, str, int, int]:
    """Return (sha256, phash hex, width, height). Raises on unreadable images."""
    import imagehash
    from PIL import Image

    sha = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        w, h = img.size
        ph = str(imagehash.phash(img.convert("RGB")))
    return sha, ph, w, h


# ---------------------------------------------------------------------------
# iNaturalist client
# ---------------------------------------------------------------------------

@dataclass
class INatClient:
    api: str = "https://api.inaturalist.org/v1"
    delay_s: float = 1.0
    user_agent: str = "SnakeCheck-dataset-builder/1.0"
    _last: float = field(default=0.0, repr=False)

    def _session(self):
        import requests
        if not hasattr(self, "_s"):
            self._s = requests.Session()
            self._s.headers["User-Agent"] = self.user_agent
        return self._s

    def _wait(self):
        dt = time.monotonic() - self._last
        if dt < self.delay_s:
            time.sleep(self.delay_s - dt)
        self._last = time.monotonic()

    def get(self, path: str, params: dict) -> dict:
        import requests
        for attempt in range(5):
            self._wait()
            try:
                r = self._session().get(f"{self.api}/{path}", params=params, timeout=60)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == 4:
                    raise
                wait = 5 * 2 ** attempt
                print(f"    API error ({e}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def download(self, url: str) -> bytes | None:
        import requests
        for attempt in range(3):
            try:
                r = self._session().get(url, timeout=30)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        return None
