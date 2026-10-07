"""Locate reviewed source in an installed checkout or its organized public export."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src/your_time" if (REPO_ROOT / "src/your_time").is_dir() else REPO_ROOT


def source_file(name: str) -> Path:
    return SOURCE_ROOT / name
