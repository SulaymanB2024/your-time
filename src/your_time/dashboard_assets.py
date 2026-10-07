"""Load reviewed dashboard source assets; rendering still embeds them offline."""

from functools import lru_cache
from pathlib import Path

ASSET_NAMES = frozenset({"dashboard.css", "dashboard.js", "contexts.js", "inspector.js"})
_SOURCE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = (_SOURCE_ROOT.parents[1] if _SOURCE_ROOT.name == "your_time"
              and _SOURCE_ROOT.parent.name == "src" else _SOURCE_ROOT)
ASSET_ROOT = _REPO_ROOT / "web"


@lru_cache(maxsize=len(ASSET_NAMES))
def read_asset(name: str) -> str:
    if name not in ASSET_NAMES:
        raise ValueError("Unknown dashboard source asset")
    return (ASSET_ROOT / name).read_text(encoding="utf-8")
