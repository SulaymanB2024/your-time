"""Index private screenshot feature prints with Apple's on-device Vision framework."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import CoreML
import Vision
from Foundation import NSURL, NSProcessInfo

from private_io import open_private_file, write_json
from secure_store import DB_PATH, STATE_DIR, connect, prepare_private_dir
from vision_batch import resource_gate

SCREENSHOT_ROOTS = (STATE_DIR / "screenshots", STATE_DIR / "pensieve/screenshots")
STATUS_PATH = STATE_DIR / "screen-feature-latest-receipt.json"
LOCK_PATH = STATE_DIR / "screen-feature-index.lock"
REVISION = 2
LEGACY_CPU_FLAG = NSProcessInfo.processInfo().operatingSystemVersion()[0] == 14


def private_screenshot(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".webp":
        return False
    if path.stat().st_uid != os.getuid():
        return False
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in SCREENSHOT_ROOTS)


def featureprint(path: Path) -> tuple[bytes, int]:
    request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    request.setRevision_(REVISION)
    # These small background fingerprints do not need the GPU/Neural Engine.
    # The macOS 14 virtual runner needs the older CPU-only flag to initialize
    # Vision; selecting a CPU stage alone still returned internal error 9.
    if LEGACY_CPU_FLAG:
        request.setUsesCPUOnly_(True)
    else:
        devices, device_error = request.supportedComputeStageDevicesAndReturnError_(None)
        if device_error or not devices:
            raise ValueError("Local Vision CPU devices unavailable")
        for stage, options in devices.items():
            cpu = next((item for item in options
                        if item.isKindOfClass_(CoreML.MLCPUComputeDevice)), None)
            if cpu is None:
                raise ValueError("Local Vision CPU device unavailable")
            request.setComputeDevice_forComputeStage_(cpu, stage)
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(str(path)), {})
    success, error = handler.performRequests_error_([request], None)
    results = request.results() or []
    if not success or error or len(results) != 1:
        # A numeric framework code is safe for synthetic CI diagnostics. Never
        # include NSError descriptions, which can contain private file paths.
        code = int(error.code()) if error else None
        raise ValueError(f"Local Vision feature print unavailable (code={code})")
    observation = results[0]
    if observation.elementType() != 1 or observation.elementCount() != 768:
        raise ValueError("Unexpected feature print format "
                         f"(type={int(observation.elementType())}, "
                         f"count={int(observation.elementCount())})")
    vector = bytes(observation.data())
    if len(vector) != 768 * 4:
        raise ValueError("Unexpected feature print length")
    return vector, int(observation.elementCount())


def distance(first: bytes, second: bytes) -> float:
    if len(first) != len(second) or len(first) != 768 * 4:
        raise ValueError("Feature print shape mismatch")
    return sum((a[0] - b[0]) ** 2 for a, b in zip(
        struct.iter_unpack("<f", first), struct.iter_unpack("<f", second))) ** 0.5


def pending_rows(limit: int) -> tuple[list[tuple[str, str]], int]:
    database = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        total = database.execute(
            "SELECT COUNT(*) FROM screenshots s LEFT JOIN screen_features f ON f.path=s.path "
            "WHERE s.ocr_status='complete' AND f.path IS NULL"
        ).fetchone()[0]
        rows = database.execute(
            "SELECT s.path,s.timestamp_utc FROM screenshots s "
            "LEFT JOIN screen_features f ON f.path=s.path "
            "WHERE s.ocr_status='complete' AND f.path IS NULL "
            "ORDER BY s.timestamp_utc DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        database.close()
    return rows, total


def write_receipt(value: dict) -> None:
    write_json(STATUS_PATH, {"checked_at_utc": datetime.now(timezone.utc).isoformat(), **value})


def run(limit: int, *, allow_battery: bool = False) -> dict:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be 1-500")
    gate = resource_gate(benchmark_now=allow_battery)
    if gate:
        return {"status": "resource_gate", "stop_reason": gate,
                "selected": 0, "completed": 0, "failed": 0}
    with connect():
        pass
    rows, pending_before = pending_rows(limit)
    started = time.monotonic()
    completed = failed = 0
    stop_reason = None
    for path_text, timestamp in rows:
        if time.monotonic() - started > 12 * 60:
            stop_reason = "time_limit"
            break
        path = Path(path_text)
        if not private_screenshot(path):
            failed += 1
            continue
        try:
            vector, element_count = featureprint(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with connect() as database:
                database.execute(
                    "INSERT OR REPLACE INTO screen_features VALUES (?, ?, ?, ?, ?, ?)",
                    (path_text, digest, REVISION, element_count, sqlite3.Binary(vector),
                     datetime.now(timezone.utc).isoformat()),
                )
            completed += 1
        except Exception:
            failed += 1
    return {"status": "complete" if not failed and not stop_reason else "partial",
            "selected": len(rows), "completed": completed, "failed": failed,
            "pending_before": pending_before,
            "pending_after": max(0, pending_before - completed),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "stop_reason": stop_reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--allow-battery", action="store_true", help="Manual bounded backfill")
    args = parser.parse_args()
    os.umask(0o077)
    prepare_private_dir()
    fd = open_private_file(LOCK_PATH)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "another_feature_index_is_running"}))
            return
        result = run(args.limit, allow_battery=args.allow_battery)
        write_receipt(result)
        print(json.dumps(result, sort_keys=True))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
