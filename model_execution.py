"""Give every local model process one private, crash-resilient inference lock."""

from __future__ import annotations

import fcntl
import math
import os
import subprocess
import sys
from pathlib import Path

from inference_telemetry import Attempt
from private_io import open_private_file


class ModelBusy(RuntimeError):
    pass


def run_model(command: list[str], *, state_dir: Path, telemetry: dict | None = None, **kwargs):
    timeout = kwargs.pop("timeout", None)
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive and finite")
    check = kwargs.pop("check", False)
    input_data = kwargs.pop("input", None)
    if kwargs.pop("capture_output", False):
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout/stderr cannot be combined with capture_output")
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if input_data is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin cannot be combined with input")
        kwargs["stdin"] = subprocess.PIPE
    fd = open_private_file(state_dir / "local-model-execution.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModelBusy("local_model_busy") from error
        # The child inherits the lock. If its coordinator exits unexpectedly,
        # the running model still excludes another model until it exits.
        attempt = Attempt(state_dir, command, telemetry)
        try:
            launch = ([sys.executable, "-B", str(Path(__file__).with_name("model_deadline.py")),
                       str(timeout), *command] if timeout is not None else command)
            with subprocess.Popen(launch, pass_fds=(fd,), **kwargs) as child:
                attempt.start_sampling(child.pid)
                try:
                    stdout, stderr = child.communicate(input_data, timeout=timeout)
                except subprocess.TimeoutExpired:
                    child.kill()
                    stdout, stderr = child.communicate()
                    attempt.finish("timeout", stderr=stderr, returncode=child.returncode)
                    error = subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
                    error.telemetry_attempt_id = attempt.identity
                    raise error from None
                except BaseException:
                    child.kill()
                    child.wait()
                    attempt.finish("interrupted", returncode=child.returncode)
                    raise
                attempt.finish("complete" if child.returncode == 0 else "process_error",
                               stderr=stderr, returncode=child.returncode)
                result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
                result.telemetry_attempt_id = attempt.identity
                result.telemetry_available = not attempt.error
                if check:
                    result.check_returncode()
                return result
        except OSError:
            attempt.finish("launch_error")
            raise
    finally:
        os.close(fd)
