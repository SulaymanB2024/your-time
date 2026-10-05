"""Give every local model process one private, crash-resilient inference lock."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat
import subprocess


class ModelBusy(RuntimeError):
    pass


def run_model(command: list[str], *, state_dir: Path, **kwargs):
    fd = os.open(state_dir / "local-model-execution.lock",
                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("Local model lock is not a private owned regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModelBusy("local_model_busy") from error
        # The child inherits the lock. If its coordinator exits unexpectedly,
        # the running model still excludes another model until it exits.
        return subprocess.run(command, pass_fds=(fd,), **kwargs)
    finally:
        os.close(fd)
