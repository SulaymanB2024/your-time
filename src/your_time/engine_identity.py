"""Fingerprint installed inference files without recording process arguments."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=32)
def file_hash(path: str, size: int, modified_ns: int) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def llama_identity(binary: Path) -> str | None:
    if not binary.is_file():
        return None
    files = [binary]
    files.extend(sorted((binary.parent.parent / "lib").glob("*.dylib")))
    files.extend(sorted(Path("/opt/homebrew/opt/ggml/lib").glob("*.dylib")))
    files.extend(sorted(Path("/opt/homebrew/opt/ggml/lib").glob("*.metallib")))
    files.extend(sorted(Path("/opt/homebrew/opt/ggml/lib").glob("*.metal")))
    records, seen = [], set()
    for entry in files:
        resolved = entry.resolve(strict=True)
        if resolved in seen:
            continue
        seen.add(resolved)
        info = resolved.stat()
        records.append({"name": resolved.name, "sha256": file_hash(str(resolved), info.st_size, info.st_mtime_ns)})
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
