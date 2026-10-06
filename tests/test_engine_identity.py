from pathlib import Path

import engine_identity


def test_engine_identity_changes_when_binary_or_local_dependency_changes(tmp_path, monkeypatch):
    binary = tmp_path / "bin/engine"
    binary.parent.mkdir()
    binary.write_bytes(b"engine1")
    library = tmp_path / "lib/libengine.dylib"
    library.parent.mkdir()
    library.write_bytes(b"library1")
    original = Path.glob

    def glob(path, pattern):
        return iter(()) if str(path).startswith("/opt/homebrew") else original(path, pattern)

    monkeypatch.setattr(Path, "glob", glob)
    first = engine_identity.llama_identity(binary)
    library.write_bytes(b"library2")
    assert engine_identity.llama_identity(binary) != first
    assert engine_identity.llama_identity(tmp_path / "absent") is None
