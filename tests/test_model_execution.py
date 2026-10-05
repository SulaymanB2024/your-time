import fcntl
import os
import subprocess
import sys

import pytest

from model_execution import ModelBusy, run_model


def test_busy_lock_prevents_a_second_process_and_releases_cleanly(tmp_path):
    fd = os.open(tmp_path / "local-model-execution.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ModelBusy):
            run_model([sys.executable, "-c", "raise RuntimeError('must not start')"], state_dir=tmp_path)
    finally:
        os.close(fd)
    result = run_model([sys.executable, "-c", "pass"], state_dir=tmp_path, timeout=5)
    assert result.returncode == 0
    assert (tmp_path / "local-model-execution.lock").stat().st_mode & 0o777 == 0o600


def test_child_keeps_lock_after_coordinator_closes_its_descriptor(tmp_path):
    fd = os.open(tmp_path / "local-model-execution.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profile = __import__("pathlib").Path(__file__).resolve().parents[1] / "network-off.sb"
    child = subprocess.Popen(["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable,
                              "-c", "import time; time.sleep(.5)"], pass_fds=(fd,))
    os.close(fd)
    try:
        with pytest.raises(ModelBusy):
            run_model([sys.executable, "-c", "pass"], state_dir=tmp_path)
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
    assert run_model([sys.executable, "-c", "pass"], state_dir=tmp_path, timeout=5).returncode == 0


def test_symlink_and_nonprivate_lock_fail_closed(tmp_path):
    target = tmp_path / "target"
    target.write_text("preserve")
    lock = tmp_path / "local-model-execution.lock"
    lock.symlink_to(target)
    with pytest.raises(OSError):
        run_model([sys.executable, "-c", "pass"], state_dir=tmp_path)
    assert target.read_text() == "preserve"
    lock.unlink()
    lock.write_text("")
    lock.chmod(0o644)
    with pytest.raises(RuntimeError, match="private owned"):
        run_model([sys.executable, "-c", "pass"], state_dir=tmp_path)
