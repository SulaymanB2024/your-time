import sqlite3
from pathlib import Path
from urllib.parse import quote

import pytest

import secure_store


def test_default_runtime_paths_use_the_synthetic_home(synthetic_home):
    assert Path.home() == synthetic_home
    assert secure_store.STATE_DIR.is_relative_to(synthetic_home)
    assert secure_store.DB_PATH.is_relative_to(synthetic_home)


def test_live_private_file_access_is_rejected_before_open(live_library_path):
    with pytest.raises(RuntimeError, match="synthetic fixtures"):
        live_library_path.read_bytes()


@pytest.mark.parametrize("uri", [False, True])
def test_live_sqlite_access_is_rejected_before_connect(live_library_path, uri):
    target = f"file:{quote(str(live_library_path))}?mode=ro" if uri else live_library_path
    with pytest.raises(RuntimeError, match="synthetic fixtures"):
        sqlite3.connect(target, uri=uri)
