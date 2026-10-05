"""Stop the network-isolated screenshot recorder before disk pressure."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from private_io import prepare_directory, write_json

STATE_DIR = Path.home() / "Library/Application Support/personal-activity-ledger"
STATUS_PATH = STATE_DIR / "screen-storage-status.json"
RECORDER_LABEL = "com.sulayman.secure-screen-record"
CONTROL_SCRIPT = "/Users/sulaymanbowles/Projects/personal-activity-ledger/capture_control.zsh"
STOP_BELOW_BYTES = 10 * 1024**3


def should_stop(free_bytes: int, running: bool, agents_enabled: bool) -> bool:
    return free_bytes < STOP_BELOW_BYTES and (running or agents_enabled)


def recorder_agent_enabled() -> bool:
    result = subprocess.run(
        ["/bin/launchctl", "print", f"gui/{os.getuid()}/{RECORDER_LABEL}"],
        check=False,
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


def recorder_pid() -> int | None:
    result = subprocess.run(
        ["/bin/ps", "-A", "-o", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    target = "/Users/sulaymanbowles/Projects/personal-activity-ledger/secure_capture.py"
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[1].startswith(
            "/Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python "
        ) and target in fields[1]:
            return int(fields[0])
    return None


def main() -> None:
    prepare_directory(STATE_DIR)
    free = shutil.disk_usage(STATE_DIR).free
    below = free < STOP_BELOW_BYTES
    pid = recorder_pid()
    running = pid is not None
    agents_enabled = recorder_agent_enabled()
    stopped = False
    if should_stop(free, running, agents_enabled):
        # Move only this task's capture agents out of LaunchAgents so they
        # cannot restart at the next login while free space remains low.
        if agents_enabled:
            subprocess.run(
                ["/bin/zsh", CONTROL_SCRIPT, "pause"],
                check=True,
                timeout=10,
                capture_output=True,
            )
        elif pid is not None:
            os.kill(pid, signal.SIGTERM)
        stopped = True
    status = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "free_bytes": free,
        "stop_below_bytes": STOP_BELOW_BYTES,
        "below_threshold": below,
        "recorder_was_running": running,
        "recorder_agent_was_enabled": agents_enabled,
        "stopped_this_run": stopped,
    }
    write_json(STATUS_PATH, status)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
