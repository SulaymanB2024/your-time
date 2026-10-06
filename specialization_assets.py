"""Download one revision of the existing 9B family; never load it here."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

from private_io import atomic_write, open_private_file, prepare_directory, write_json
from secure_store import STATE_DIR

REPOSITORY = "mlx-community/Qwen3.5-9B-MLX-4bit"
REVISION = "938d8919941c6e7efd3c7150eff7fe9d12afa631"
MODEL_DIR = STATE_DIR / "models/qwen3.5-9b-mlx-4bit"
ASSET_MANIFEST = STATE_DIR / "specialization/mlx-asset-manifest.json"
WEIGHTS = {
    "model-00001-of-00002.safetensors": (5349771222, "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd"),
    "model-00002-of-00002.safetensors": (600449850, "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d"),
}
SMALL_FILES = ("config.json", "chat_template.jinja", "model.safetensors.index.json",
               "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
               "tokenizer.json", "video_preprocessor_config.json", "vocab.json")
FLOOR = int(10.5 * 1024**3)


def digest(path: Path) -> str:
    fd = open_private_file(path, os.O_RDONLY)
    h = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> dict:
    prepare_directory(MODEL_DIR)
    records = []
    for name in (*SMALL_FILES, *WEIGHTS):
        target = MODEL_DIR / name
        expected_size, expected_sha = WEIGHTS.get(name, (None, None))
        if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
            actual_sha = digest(target)
            if expected_sha is None or actual_sha == expected_sha:
                records.append({"name": name, "size": target.stat().st_size, "sha256": actual_sha})
                continue
        remaining = sum(size for filename, (size, _) in WEIGHTS.items() if not (MODEL_DIR / filename).exists())
        if shutil.disk_usage(MODEL_DIR).free < FLOOR + remaining:
            raise RuntimeError("download_disk_reserve")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            if expected_size is None:
                payload = response.read(32 * 1024**2 + 1)
                if len(payload) > 32 * 1024**2:
                    raise RuntimeError("model_metadata_too_large")
                atomic_write(target, payload)
            else:
                partial = MODEL_DIR / (name + ".partial")
                fd = open_private_file(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                with os.fdopen(fd, "wb") as stream:
                    count = 0
                    while chunk := response.read(8 * 1024**2):
                        count += len(chunk)
                        if count > expected_size or shutil.disk_usage(MODEL_DIR).free < FLOOR:
                            raise RuntimeError("download_size_or_disk_reserve")
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if count != expected_size or digest(partial) != expected_sha:
                    raise RuntimeError("model_asset_hash_mismatch")
                # The destination is checked by the same private-file helper.
                if target.exists():
                    os.close(open_private_file(target, os.O_RDONLY))
                os.replace(partial, target)
        records.append({"name": name, "size": target.stat().st_size, "sha256": digest(target)})
    manifest = {"version": "mlx_assets_v1", "repository": REPOSITORY,
                "revision": REVISION, "files": records}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(ASSET_MANIFEST, manifest)
    return {"status": "downloaded_verified", "files": len(records),
            "bytes": sum(r["size"] for r in records), "manifest_sha256": manifest["manifest_sha256"]}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(json.dumps(download()))
    except Exception as error:
        # Do not expose signed CDN URLs or response bodies in command output.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
