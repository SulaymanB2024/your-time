"""Give every local model process one private, crash-resilient inference lock."""

from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path

from private_io import open_private_file


class ModelBusy(RuntimeError):
    pass


def run_model(command: list[str], *, state_dir: Path, **kwargs):
    fd = open_private_file(state_dir / "local-model-execution.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModelBusy("local_model_busy") from error
        # The child inherits the lock. If its coordinator exits unexpectedly,
        # the running model still excludes another model until it exits.
        return subprocess.run(command, pass_fds=(fd,), **kwargs)
    finally:
        os.close(fd)
