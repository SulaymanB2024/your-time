"""Load reviewed dashboard source assets; rendering still embeds them offline."""

from functools import lru_cache
from pathlib import Path

ASSET_NAMES = frozenset({"dashboard.css", "dashboard.js", "contexts.js", "inspector.js"})
ASSET_ROOT = Path(__file__).resolve().parent / "web"


@lru_cache(maxsize=len(ASSET_NAMES))
def read_asset(name: str) -> str:
    if name not in ASSET_NAMES:
        raise ValueError("Unknown dashboard source asset")
    return (ASSET_ROOT / name).read_text(encoding="utf-8")
