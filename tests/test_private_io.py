"""Exercise filesystem escape, failure recovery and concurrency boundaries."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

import daily_export
import phone_import
import private_io
import vision_batch


def test_legacy_temporary_symlink_cannot_overwrite_an_unrelated_file(tmp_path):
    victim = tmp_path / "unrelated"
    victim.write_bytes(b"preserve")
    root = tmp_path / "outputs"
    root.mkdir(mode=0o700)
    output = root / "day.jsonl"
    old_temp = output.with_suffix(".jsonl.tmp")
    old_temp.symlink_to(victim)
    daily_export.private_write(output, b"synthetic private export")
    assert victim.read_bytes() == b"preserve"
    assert output.read_bytes() == b"synthetic private export"
    assert not output.is_symlink()
    assert old_temp.is_symlink()


@pytest.mark.parametrize("writer", [
    lambda path: private_io.atomic_write(path, b"new"),
    lambda path: vision_batch.write_receipt({"synthetic": True}, path),
    lambda path: phone_import.write_private_json(path, {"synthetic": True}),
])
@pytest.mark.parametrize("link_type", ["symbolic", "hard"])
def test_linked_destination_fails_closed_without_changing_victim(tmp_path, writer, link_type):
    victim = tmp_path / "victim"
    victim.write_bytes(b"preserve")
    victim.chmod(0o600)
    output = tmp_path / "output"
    if link_type == "symbolic":
        output.symlink_to(victim)
    else:
        os.link(victim, output)
    with pytest.raises((OSError, RuntimeError)):
        writer(output)
    assert victim.read_bytes() == b"preserve"
    assert not list(tmp_path.glob(".private-*.tmp"))


def test_linked_parent_is_rejected_before_changing_permissions_or_contents(tmp_path):
    victim = tmp_path / "elsewhere"
    victim.mkdir(mode=0o755)
    victim.chmod(0o755)
    link = tmp_path / "alias"
    link.symlink_to(victim, target_is_directory=True)
    with pytest.raises(OSError):
        private_io.atomic_write(link / "nested" / "output", b"new")
    assert victim.stat().st_mode & 0o777 == 0o755
    assert list(victim.iterdir()) == []


def test_failed_write_preserves_last_good_receipt_and_removes_temporary(tmp_path, monkeypatch):
    output = tmp_path / "receipt.json"
    private_io.write_json(output, {"old": True})
    monkeypatch.setattr(private_io.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("synthetic failure")))
    with pytest.raises(OSError, match="synthetic failure"):
        private_io.write_json(output, {"new": True})
    assert json.loads(output.read_bytes()) == {"old": True}
    assert not list(tmp_path.glob(".private-*.tmp"))


def test_concurrent_writers_publish_only_whole_private_payloads(tmp_path):
    output = tmp_path / "receipt.json"
    payloads = [(json.dumps({"row": n, "text": str(n) * 10000}) + "\n").encode() for n in range(24)]
    private_io.atomic_write(output, payloads[0])

    def publish(payload):
        private_io.atomic_write(output, payload)
        assert output.read_bytes() in payloads

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(publish, payloads))
    assert output.read_bytes() in payloads
    assert output.stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert not list(tmp_path.glob(".private-*.tmp"))


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "nonprivate", "fifo"])
def test_lock_file_rejects_unsafe_types_without_modifying_target(tmp_path, kind):
    path = tmp_path / "lock"
    victim = tmp_path / "victim"
    victim.write_bytes(b"preserve")
    victim.chmod(0o600)
    if kind == "symlink":
        path.symlink_to(victim)
    elif kind == "hardlink":
        os.link(victim, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    else:
        path.write_bytes(b"preserve")
        path.chmod(0o644)
    with pytest.raises((OSError, RuntimeError)):
        private_io.open_private_file(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    assert victim.read_bytes() == b"preserve"
    if kind == "nonprivate":
        assert path.read_bytes() == b"preserve"
