"""Download the public ARC-AGI-1 training set into ``backend/data/arc1``.

The data ships with the repository, so this is only needed to refresh it or to
rebuild the folder from scratch. Standard library only.

    python backend/scripts/fetch_arc1_data.py
"""

from __future__ import annotations

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
TARGET = Path(__file__).resolve().parents[1] / "data" / "arc1"


def main() -> int:
    print(f"Downloading {SOURCE} ...")
    with urllib.request.urlopen(SOURCE, timeout=120) as response:  # noqa: S310 - fixed URL
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    training = TARGET / "training"
    if training.exists():
        shutil.rmtree(training)
    training.mkdir(parents=True)

    count = 0
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[1:3] == ["data", "training"] and parts[3].endswith(".json"):
            (training / parts[3]).write_bytes(archive.read(name))
            count += 1
        elif len(parts) == 2 and parts[1] == "LICENSE":
            (TARGET / "LICENSE").write_bytes(archive.read(name))

    print(f"Wrote {count} tasks to {training}")
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
