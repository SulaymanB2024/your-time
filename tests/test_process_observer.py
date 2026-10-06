import ctypes
import hashlib
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import process_observer as observer


def item(pid=10, parent=1, start=100, ticks=0, ns=0):
    return {"pid": pid, "ppid": parent, "start_abstime": start,
            "cpu_ticks": ticks, "monotonic_ns": ns,
            "rss_bytes": 1024, "physical_footprint_bytes": 2048}


def test_mach_ticks_are_converted_and_logical_core_cpu_is_separate():
    old = item()
    current = item(ticks=24_000_000, ns=1_000_000_000)
    row = observer.interval_rows([current], {(10, 100): old}, 125, 3, 8)[0]
    assert row["cpu_percent_one_core_interval"] == 100
    assert row["cpu_percent_all_logical_cores_interval"] == 12.5
    assert row["rss_bytes"] == 1024
    assert row["physical_footprint_bytes"] == 2048


@pytest.mark.parametrize("change", [
    {"start_abstime": 101}, {"ppid": 2}, {"monotonic_ns": 0}, {"cpu_ticks": 0}])
def test_first_sample_reuse_reparenting_or_regressing_counters_are_unknown(change):
    old = item(ticks=10, ns=1)
    current = {**item(ticks=20, ns=1_000_000_001), **change}
    rows = observer.interval_rows([current], {(10, 100): old}, 125, 3, 8)
    assert rows[0]["cpu_percent_one_core_interval"] is None
    assert rows[0]["interval_seconds"] is None
    assert observer.interval_rows([current], {}, 125, 3, 8)[0]["cpu_percent_one_core_interval"] is None


def test_native_child_inventory_uses_count_not_bytes_and_fails_on_truncation():
    native = observer.Native.__new__(observer.Native)
    def children(pid, array, size):
        array[0], array[1] = 11, 12
        return 2
    native.lib = SimpleNamespace(proc_listchildpids=children)
    assert native.children(10) == [11, 12]
    native.lib.proc_listchildpids = lambda *args: observer.MAX_PROCESSES + 1
    with pytest.raises(observer.ObserverError, match="process_scope_limit"):
        native.children(10)


def test_native_foreign_uid_is_rejected_without_returning_names():
    native = observer.Native.__new__(observer.Native)
    def info(pid, flavor, arg, buffer, size):
        record = ctypes.cast(buffer, ctypes.POINTER(observer.ShortInfo)).contents
        record.pid, record.uid = pid, os.getuid() + 1
        record.ignored_name = b"PRIVATE-NAME"
        return size
    native.lib = SimpleNamespace(proc_pidinfo=info)
    assert native.observation(10) is None


class FakeNative:
    numer, denom = 125, 3
    def __init__(self):
        self.nodes = {10: item(), 11: item(11, 10), 12: item(12, 99)}
    def observation(self, pid):
        return self.nodes.get(pid)
    def children(self, pid):
        return [11, 12] if pid == 10 else []


def test_tree_does_not_accept_wrong_parent_or_reused_job_root():
    native = FakeNative()
    assert {r["pid"] for r in observer.tree(native, item())} == {10, 11}
    native.nodes[10] = item(start=101)
    with pytest.raises(observer.ObserverError, match="root_exited_or_reused"):
        observer.tree(native, item())


def test_reparented_branch_is_removed_and_missing_root_is_not_a_success():
    native = FakeNative()
    native.nodes[12] = item(12, 11)
    native.children = lambda pid: {10: [11], 11: [12]}.get(pid, [])
    calls = [0]
    base = native.observation
    def reparent(pid):
        if pid == 11:
            calls[0] += 1
            if calls[0] >= 2:
                return item(11, 999)
        return base(pid)
    native.observation = reparent
    assert {r["pid"] for r in observer.tree(native, item())} == {10}
    native.observation = lambda pid: None
    with pytest.raises(observer.ObserverError, match="root_counters_unavailable"):
        observer.tree(native, item())


def controlled_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(observer.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(observer.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    monkeypatch.setattr(observer, "job_root", lambda: 10)
    monkeypatch.setattr(observer, "remaining_seconds", lambda *a, **kw: 100)
    return clock


def test_bounded_observer_retains_only_counters_private_hash_verified_spool(tmp_path, monkeypatch):
    controlled_clock(monkeypatch)
    value = observer.observe(tmp_path, max_seconds=2, interval=1, native=FakeNative())
    assert value["status"] == "complete"
    assert value["stop_reason"] == "observer_deadline"
    assert value["samples"] == 2
    assert value["attempt_linkage"].startswith("none;")
    spool = tmp_path / "process-observations" / value["sample_file"]
    assert hashlib.sha256(spool.read_bytes()).hexdigest() == value["sample_sha256"]
    assert spool.stat().st_mode & 0o777 == 0o600
    fields = json.loads(spool.read_text().splitlines()[0])["processes"][0]
    assert set(fields) == {"pid", "ppid", "start_abstime", "cpu_percent_one_core_interval",
                          "cpu_percent_all_logical_cores_interval", "interval_seconds",
                          "rss_bytes", "physical_footprint_bytes"}
    assert "PRIVATE-NAME" not in spool.read_text()


def test_zero_budget_window_and_changed_root_do_not_watch_later_job(tmp_path, monkeypatch):
    controlled_clock(monkeypatch)
    monkeypatch.setattr(observer, "remaining_seconds", lambda *a, **kw: 0)
    assert observer.observe(tmp_path, max_seconds=2, native=FakeNative())["status"] == "skipped"
    monkeypatch.setattr(observer, "remaining_seconds", lambda *a, **kw: 100)
    calls = iter([10, 999])
    monkeypatch.setattr(observer, "job_root", lambda: next(calls))
    value = observer.observe(tmp_path, max_seconds=2, native=FakeNative())
    assert value["stop_reason"] == "job_exited_or_changed"
    assert value["samples"] == 0


def test_observer_storage_and_singleton_bounds(tmp_path, monkeypatch):
    controlled_clock(monkeypatch)
    monkeypatch.setattr(observer, "MAX_BYTES", 1)
    assert observer.observe(tmp_path, max_seconds=2, native=FakeNative())["stop_reason"] == "observer_storage_limit"
    import fcntl

    from private_io import open_private_file
    fd = open_private_file(tmp_path / "process-observations/observer.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = observer.observe(tmp_path, max_seconds=2, native=FakeNative())
        assert result["stop_reason"] == "observer_already_running"
    finally:
        os.close(fd)


def test_ten_second_synthesis_margin_is_reserved(tmp_path, monkeypatch):
    controlled_clock(monkeypatch)
    monkeypatch.setattr(observer, "remaining_seconds", lambda *a, **kw: 12)
    result = observer.observe(tmp_path, max_seconds=50, interval=1, native=FakeNative())
    assert result["elapsed_seconds"] == 2
    start = datetime.fromisoformat(result["started_at_utc"])
    cutoff = datetime.fromisoformat(result["cutoff_at_utc"])
    assert start.tzinfo == timezone.utc and (cutoff - start).total_seconds() == 2


def test_partial_write_failure_retains_exact_bytes_hash_and_explicit_failure(tmp_path, monkeypatch):
    controlled_clock(monkeypatch)
    original = os.write
    writes = [0]
    def partial(fd, body):
        writes[0] += 1
        if writes[0] == 1:
            return original(fd, body[:13])
        raise OSError("PRIVATE PATH MUST NOT ENTER RECEIPT")
    monkeypatch.setattr(observer.os, "write", partial)
    result = observer.observe(tmp_path, max_seconds=2, native=FakeNative())
    spool = tmp_path / "process-observations" / result["sample_file"]
    assert result["status"] == "failed" and result["stop_reason"] == "observer_failed"
    assert result["trailing_record_incomplete"]
    assert result["samples"] == 0 and result["sample_bytes"] == 13
    assert hashlib.sha256(spool.read_bytes()).hexdigest() == result["sample_sha256"]
    assert "PRIVATE PATH" not in json.dumps(result)


def test_wake_after_clock_window_closes_does_not_resume_observation(tmp_path, monkeypatch):
    controlled_clock(monkeypatch)
    remaining = iter([100, 100, 0])
    monkeypatch.setattr(observer, "remaining_seconds", lambda *a, **kw: next(remaining))
    result = observer.observe(tmp_path, max_seconds=50, interval=1, native=FakeNative())
    assert result["status"] == "complete" and result["stop_reason"] == "observer_deadline"
    assert result["samples"] == 1 and result["elapsed_seconds"] == 1
