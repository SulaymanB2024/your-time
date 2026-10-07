"""Download pinned public vision weights for a private, bounded comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from private_io import prepare_directory
from secure_store import STATE_DIR, prepare_private_dir

CANDIDATES = {
    "9b": (Path(__file__).with_name("vision_quality_model_manifest.json"),
           STATE_DIR / "models/qwen3.5-9b-q8-eval"),
    "27b": (Path(__file__).with_name("vision_peak_model_manifest.json"),
            STATE_DIR / "models/qwen3.5-27b-q3-eval"),
}
MIN_FREE_BYTES = 20 * 1024**3
MAX_DOWNLOAD_SECONDS = 90 * 60


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def on_ac_power() -> bool:
    result = subprocess.run(["/usr/bin/pmset", "-g", "batt"], check=True,
                            capture_output=True, text=True, timeout=10)
    return "AC Power" in result.stdout.splitlines()[0]


def download(item: dict, repository: str, revision: str, target_dir: Path,
             *, allow_battery: bool) -> None:
    name = item["name"]
    target = target_dir / name
    if target.exists():
        if target.is_symlink() or target.stat().st_size != item["size"] or digest(target) != item["sha256"]:
            raise RuntimeError(f"Existing model file failed verification: {name}")
        print(f"verified_existing {name}", flush=True)
        return
    partial = target_dir / (name + ".partial")
    if partial.exists() and (partial.is_symlink() or partial.stat().st_size > item["size"]):
        raise RuntimeError(f"Invalid partial model file: {name}")
    remaining = item["size"] - (partial.stat().st_size if partial.exists() else 0)
    if shutil.disk_usage(target_dir).free - remaining < MIN_FREE_BYTES:
        raise RuntimeError("Insufficient disk space above the 20 GiB safety floor")
    if not allow_battery and not on_ac_power():
        raise RuntimeError("Connect AC power before downloading the quality model")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
    command = ["/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
               "--retry", "4", "--retry-all-errors", "--connect-timeout", "20",
               "--proto", "=https", "--proto-redir", "=https", "--continue-at", "-",
               "--output", str(partial), url]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    last_reported = partial.stat().st_size if partial.exists() else 0
    while process.poll() is None:
        time.sleep(5)
        if (shutil.disk_usage(target_dir).free < MIN_FREE_BYTES
                or (not allow_battery and not on_ac_power())
                or time.monotonic() - started > MAX_DOWNLOAD_SECONDS):
            process.terminate()
            process.wait(timeout=10)
            raise RuntimeError(f"Stopped quality-model download at safety gate: {name}")
        current = partial.stat().st_size if partial.exists() else 0
        if current - last_reported >= 256 * 1024**2:
            print(f"downloaded {name} {current}/{item['size']} bytes", flush=True)
            last_reported = current
    if process.returncode:
        raise RuntimeError(f"Model download failed with status {process.returncode}: {name}")
    if partial.stat().st_size != item["size"] or digest(partial) != item["sha256"]:
        raise RuntimeError(f"Downloaded model failed size/SHA-256 verification: {name}")
    os.chmod(partial, 0o400)
    os.replace(partial, target)
    print(f"verified_download {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="9b")
    parser.add_argument("--allow-battery", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    manifest_path, target_dir = CANDIDATES[args.candidate]
    config = json.loads(manifest_path.read_text())
    prepare_private_dir()
    prepare_directory(target_dir)
    for item in config["files"]:
        download(item, config["repository"], config["revision"], target_dir,
                 allow_battery=args.allow_battery)


if __name__ == "__main__":
    main()
