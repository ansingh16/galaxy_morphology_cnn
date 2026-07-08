"""Fetch the Galaxy10 SDSS dataset.

Galaxy10 is 21,785 real SDSS galaxy images (69x69, RGB) hand-labelled into 10
morphology classes, put together by the astroNN project. It's small enough to
train on a laptop CPU, which is the whole point here.

    python -m galaxy_cnn.download_data

The .h5 lands in data/ (gitignored). Run once; it's ~200 MB.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "http://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5"
DEST = Path(__file__).resolve().parents[2] / "data" / "Galaxy10.h5"


def _progress(block_num: int, block_size: int, total: int) -> None:
    done = block_num * block_size
    pct = min(100, done * 100 // total) if total > 0 else 0
    sys.stdout.write(f"\r  {pct:3d}%  ({done // (1024 * 1024)} MB)")
    sys.stdout.flush()


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already have {DEST} ({DEST.stat().st_size // (1024 * 1024)} MB)")
        return
    print(f"downloading Galaxy10 -> {DEST}")
    urllib.request.urlretrieve(URL, DEST, _progress)
    print(f"\ndone: {DEST.stat().st_size // (1024 * 1024)} MB")


if __name__ == "__main__":
    main()
