"""Run source checks and synthetic tests without opening the private ledger."""

from __future__ import annotations

import argparse
import json
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True, timeout=300)


def check_source() -> None:
    from publish_source import CI_FILES, DOCS, JSON_FILES, ROOT_SOURCE_FILES, allowed

    candidates = {ROOT / name for name in CI_FILES | DOCS | JSON_FILES | ROOT_SOURCE_FILES}
    for directory in ("tests", "launchagents", "native", "browser_extension", "tools"):
        folder = ROOT / directory
        if not folder.is_dir() or folder.is_symlink():
            raise RuntimeError("Registered source directory must not be linked")
        candidates.update(folder.iterdir())
    sources = sorted(path for path in candidates
                     if allowed(path.relative_to(ROOT).as_posix()))
    for path in sources:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("Registered source must be a regular file")
    counts = {"python": 0, "shell": 0, "plist": 0, "json": 0, "javascript": 0}
    for path in sources:
        if path.suffix == ".py":
            compile(path.read_bytes(), str(path), "exec")
            counts["python"] += 1
        elif path.suffix == ".zsh":
            run(["/bin/zsh", "-n", str(path)])
            counts["shell"] += 1
        elif path.suffix == ".plist":
            plistlib.loads(path.read_bytes())
            counts["plist"] += 1
        elif path.suffix == ".json":
            json.loads(path.read_bytes())
            counts["json"] += 1
        elif path.suffix == ".js":
            run(["node", "--check", str(path)])
            counts["javascript"] += 1

    # Check the actual assembled dashboard script, including its helper chunks.
    from dashboard_ui import JS

    with tempfile.TemporaryDirectory(prefix="your-time-source-check-") as temporary:
        script = Path(temporary) / "dashboard.js"
        script.write_text(JS)
        run(["node", "--check", str(script)])
        counts["javascript"] += 1
    print(json.dumps({"source_syntax": "passed", "files": counts}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-tests", action="store_true", help="Run lint and source syntax only")
    args = parser.parse_args()
    run([sys.executable, "-m", "ruff", "check", "."])
    check_source()
    if not args.no_tests:
        run([sys.executable, "-m", "pytest", "-q"])


if __name__ == "__main__":
    main()
