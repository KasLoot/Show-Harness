"""Fetch only the official Panda MJCF and its meshes at a pinned Menagerie revision."""
from pathlib import Path
import xml.etree.ElementTree as ET

import requests


REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
DEST = Path(__file__).resolve().parents[2] / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/franka_emika_panda"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        def fetch(name: str) -> bytes:
            path = DEST / name
            if path.is_file():
                return path.read_bytes()
            response = session.get(f"{BASE}/{name}", timeout=60)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return response.content

        panda = ET.fromstring(fetch("panda.xml"))
        fetch("LICENSE")
        for mesh in panda.findall("asset/mesh"):
            name = mesh.attrib["file"]
            if Path(name).name != name:
                raise ValueError(f"Unexpected mesh path: {name!r}")
            fetch(f"assets/{name}")
    print(f"Panda assets ready: {DEST} (Menagerie {REVISION[:12]})")


if __name__ == "__main__":
    main()
