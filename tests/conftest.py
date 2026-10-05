"""Keep collection-time paths and test I/O away from the user's private state."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

import pytest

_LIVE_LIBRARY = (Path.home() / "Library").resolve()
_HOME_ENVIRONMENT = ("HOME", "CFFIXED_USER_HOME")
_saved_environment: dict[str, str | None] = {}
_test_home: tempfile.TemporaryDirectory | None = None
_guard_enabled = False


def _guard_private_access(event: str, arguments: tuple) -> None:
    if not _guard_enabled or event not in {"open", "sqlite3.connect"}:
        return
    value = arguments[0]
    if not isinstance(value, (str, bytes, os.PathLike)):
        return  # Opening an already-owned file descriptor is not path access.
    value = os.fsdecode(value)
    if event == "sqlite3.connect":
        if value == ":memory:":
            return
        if value.startswith("file:"):
            value = unquote(value[5:].split("?", 1)[0])
    path = Path(value).resolve()
    if path.is_relative_to(_LIVE_LIBRARY):
        raise RuntimeError("Tests must use synthetic fixtures, never the user's Library")


sys.addaudithook(_guard_private_access)


def pytest_configure(config) -> None:
    global _test_home, _guard_enabled
    _test_home = tempfile.TemporaryDirectory(prefix="your-time-test-home-")
    for name in _HOME_ENVIRONMENT:
        _saved_environment[name] = os.environ.get(name)
        os.environ[name] = _test_home.name
    # Runs before test modules import cached STATE_DIR/DB_PATH constants.
    _guard_enabled = True


def pytest_unconfigure(config) -> None:
    global _guard_enabled
    _guard_enabled = False
    for name, value in _saved_environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if _test_home is not None:
        _test_home.cleanup()


@pytest.fixture(scope="session")
def synthetic_home() -> Path:
    assert _test_home is not None
    return Path(_test_home.name)


@pytest.fixture
def live_library_path() -> Path:
    """An inaccessible sentinel path; tests must never create or read it."""
    return _LIVE_LIBRARY / "your-time-test-access-must-be-rejected"
