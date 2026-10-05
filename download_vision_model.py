"""Download pinned public GGUF files without sending private activity data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from secure_store import STATE_DIR, prepare_private_dir


MANIFEST = Path(__file__).with_name("vision_model_manifest.json")
MODEL_DIR = STATE_DIR / "models/qwen3.5-2b-q4km"
FLOOR_BYTES = 11 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    os.umask(0o077)
    config = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prepare_private_dir()
    MODEL_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(MODEL_DIR, 0o700)
    for item in config["files"]:
        name = item["name"]
        target = MODEL_DIR / name
        if target.exists() and target.stat().st_size == item["size"] and sha256_file(target) == item["sha256"]:
            print(f"verified_existing {name}", flush=True)
            continue
        if target.exists():
            raise RuntimeError(f"Existing model file failed integrity verification: {name}")
        if shutil.disk_usage(MODEL_DIR).free - item["size"] < FLOOR_BYTES:
            raise RuntimeError(f"Insufficient free space to download {name} while preserving 11 GiB")
        partial = MODEL_DIR / (name + ".partial")
        if partial.exists():
            raise RuntimeError(f"Partial download already exists; inspect before retry: {name}")
        url = f"https://huggingface.co/{config['repository']}/resolve/{config['revision']}/{name}"
        command = [
            "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
            "--proto", "=https", "--proto-redir", "=https", "--max-filesize", str(item["size"] + 1024),
            "--output", str(partial), url,
        ]
        process = subprocess.Popen(command)
        reported = 0
        while process.poll() is None:
            time.sleep(5)
            free = shutil.disk_usage(MODEL_DIR).free
            if free < FLOOR_BYTES:
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError("Download stopped before disk safety floor")
            current = partial.stat().st_size if partial.exists() else 0
            if current - reported >= 256 * 1024**2:
                print(f"downloaded {name} {current}/{item['size']} bytes", flush=True)
                reported = current
        if process.returncode != 0:
            raise RuntimeError(f"Download failed for {name} with exit status {process.returncode}")
        if partial.stat().st_size != item["size"] or sha256_file(partial) != item["sha256"]:
            raise RuntimeError(f"Downloaded file failed size or SHA-256 check: {name}")
        os.chmod(partial, 0o400)
        os.replace(partial, target)
        print(f"verified_download {name}", flush=True)


if __name__ == "__main__":
    main()
