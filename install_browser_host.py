"""Install the Chrome Native Messaging allowlist for one reviewed extension ID."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


NAME = "com.sulayman.personal_activity_ledger"
DESTINATION = (Path.home() / "Library/Application Support/Google/Chrome/NativeMessagingHosts"
               / f"{NAME}.json")
WRAPPER = Path(__file__).with_name("secure_browser_bridge.zsh")


def manifest(extension_id: str) -> dict:
    if not re.fullmatch(r"[a-p]{32}", extension_id):
        raise ValueError("Invalid Chrome extension ID")
    return {"name": NAME, "description": "Private Your Time browser bridge",
            "path": str(WRAPPER), "type": "stdio",
            "allowed_origins": [f"chrome-extension://{extension_id}/"]}


def install(extension_id: str) -> Path:
    value = manifest(extension_id)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(DESTINATION.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=".native-host-", dir=DESTINATION.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, DESTINATION)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return DESTINATION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension_id")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        print(json.dumps({"installed": str(install(args.extension_id))}))
    else:
        print(json.dumps(manifest(args.extension_id), indent=2))


if __name__ == "__main__":
    main()
