"""Fetch pinned Panda/RoboLab assets and the authored scene's dependency closure.

RoboLab dependency discovery uses usd-core, without loading Isaac Sim.
Assets, source definitions and license notices live under gitignored models/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import posixpath
import re
from urllib.request import urlopen
import xml.etree.ElementTree as ET

REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{REVISION}/franka_emika_panda"
)
DEFAULT_DEST = Path(__file__).resolve().parents[2] / "models/mujoco/franka_emika_panda"
ROBOLAB_REVISION = "ad45d4f974725d020f82c2b0d77d78533aeba2b3"
ROBOLAB_DEST = DEFAULT_DEST.parent / "robolab"


def download_assets(destination: Path = DEFAULT_DEST) -> Path:
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> None:
        path = destination / name
        if path.is_file() and path.stat().st_size:
            return
        with urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(path)

    marker = destination / "REVISION"
    if marker.exists() and marker.read_text().strip() != REVISION:
        raise RuntimeError(f"{destination} contains a different revision; choose a new directory.")
    fetch("panda.xml")
    root = ET.parse(destination / "panda.xml").getroot()
    files = ["LICENSE", "README.md"] + [
        f"assets/{mesh.attrib['file']}" for mesh in root.findall("asset/mesh")
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files))
    marker.write_text(REVISION + "\n")
    print(f"Panda assets ready: {destination} ({REVISION})")
    return destination


def download_robolab(destination: Path = ROBOLAB_DEST) -> Path:
    """Download the authored scene and its dependency closure, resolving Git LFS."""
    from pxr import Sdf

    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "REVISION"
    if marker.exists() and marker.read_text().strip() != ROBOLAB_REVISION:
        raise RuntimeError(f"{destination} contains a different RoboLab revision")
    pending = ["assets/scenes/rubiks_cube_bowl.usda", "LICENSE", "THIRD_PARTY_NOTICES.md",
               "robolab/tasks/benchmark/rubiks_cube_task.py",
               "robolab/core/task/conditionals.py", "robolab/core/task/predicate_logic.py",
               "robolab/core/task/hull_check.py",
               "robolab/core/scenes/utils.py"]
    seen = set()
    manifest = {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            base = f"https://raw.githubusercontent.com/NVlabs/RoboLab/{ROBOLAB_REVISION}"
            with urlopen(f"{base}/{name}", timeout=60) as response:
                content = response.read()
            if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
                pointer = content.decode().splitlines()
                digest = next(line.split(":", 1)[1] for line in pointer if line.startswith("oid "))
                size = int(next(line.split()[1] for line in pointer if line.startswith("size ")))
                base = f"https://media.githubusercontent.com/media/NVlabs/RoboLab/{ROBOLAB_REVISION}"
                with urlopen(f"{base}/{name}", timeout=90) as response:
                    content = response.read()
                if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                    raise RuntimeError(f"Git LFS verification failed: {name}")
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(content)
            partial.replace(target)
            print(f"[assets] {name}: {len(content):,} bytes", flush=True)
        content = target.read_bytes()
        manifest[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        if target.suffix in (".usd", ".usda", ".usdc"):
            layer = Sdf.Layer.FindOrOpen(str(target.resolve()))
            if layer is None:
                raise RuntimeError(f"Cannot read USD layer {target}")
            # Composition dependencies omit shader texture attributes. Exporting the
            # parsed layer captures those asset paths in binary USD layers as well.
            refs = set(re.findall(r"@([^@]+)@", layer.ExportToString()))
            for ref in refs:
                if not ref or ref.endswith(".mdl"):
                    continue  # MDL shader code is not supported by MuJoCo's renderer.
                if "://" in ref or ref.startswith("/"):
                    raise RuntimeError(f"Unmapped external USD dependency: {ref}")
                relative = posixpath.normpath(posixpath.join(posixpath.dirname(name), ref))
                if relative.startswith("../"):
                    raise RuntimeError(f"USD dependency escapes the asset root: {ref}")
                pending.append(relative)
    marker.write_text(ROBOLAB_REVISION + "\n")
    (destination / "manifest.json").write_text(json.dumps({
        "repository": "NVlabs/RoboLab", "revision": ROBOLAB_REVISION, "files": manifest
    }, indent=2) + "\n")
    print(f"RoboLab assets ready: {len(manifest)} files, "
          f"{sum(row['bytes'] for row in manifest.values()) / 1e6:.1f} MB", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    download_assets(args.destination)
    download_robolab()
