"""Bounded, content-free observations of the exact overnight job's processes.

This is supplemental history, not an amendment to frozen inference attempts.
CPU is a native interval measurement; memory is per process, never summed.
No process names, arguments, environment, activity or model output are retained.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from overnight_schedule import VISION_END, remaining_seconds
from private_io import open_private_file, write_json
from secure_store import STATE_DIR

VERSION = "supplemental_process_observer_v1"
JOB = "com.yourtime.overnight-vision"
MAX_PROCESSES = 64
MAX_BYTES = 4 * 1024**2


class ObserverError(RuntimeError):
    pass


class Usage(ctypes.Structure):
    # SDK rusage_info_v0. CPU counters are Mach ticks, not nanoseconds.
    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64) for name in
        ("user", "system", "idle", "interrupts", "pageins", "wired", "rss",
         "footprint", "start", "exit")]


class ShortInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("pid", "ppid", "pgid", "status")] + [
        ("ignored_name", ctypes.c_char * 16)] + [
        (name, ctypes.c_uint32) for name in
        ("flags", "uid", "gid", "ruid", "rgid", "svuid", "svgid", "reserved")]


class Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class Native:
    def __init__(self):
        if sys.platform != "darwin":
            raise ObserverError("native_counters_unavailable")
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.proc_pid_rusage.restype = ctypes.c_int
        self.lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                          ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_pidinfo.restype = ctypes.c_int
        self.lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_listchildpids.restype = ctypes.c_int
        clock = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        clock.mach_timebase_info.argtypes = [ctypes.POINTER(Timebase)]
        clock.mach_timebase_info.restype = ctypes.c_int
        base = Timebase()
        if clock.mach_timebase_info(ctypes.byref(base)) or not base.numer or not base.denom:
            raise ObserverError("native_timebase_unavailable")
        self.numer, self.denom = base.numer, base.denom

    def observation(self, pid):
        before, after, usage, check = ShortInfo(), ShortInfo(), Usage(), Usage()
        if (self.lib.proc_pidinfo(pid, 13, 0, ctypes.byref(before), ctypes.sizeof(before)) != ctypes.sizeof(before)
                or before.uid != os.getuid() or before.pid != pid
                or self.lib.proc_pid_rusage(pid, 0, ctypes.byref(usage))
                or self.lib.proc_pidinfo(pid, 13, 0, ctypes.byref(after), ctypes.sizeof(after)) != ctypes.sizeof(after)
                or (before.pid, before.ppid, before.uid) != (after.pid, after.ppid, after.uid)
                or self.lib.proc_pid_rusage(pid, 0, ctypes.byref(check))
                or usage.start != check.start or not usage.start or usage.exit):
            return None
        return {"pid": pid, "ppid": before.ppid, "start_abstime": int(usage.start),
                "cpu_ticks": int(usage.user + usage.system), "rss_bytes": int(usage.rss),
                "physical_footprint_bytes": int(usage.footprint),
                "monotonic_ns": time.monotonic_ns()}

    def children(self, pid):
        array = (ctypes.c_int * (MAX_PROCESSES + 1))()
        ctypes.set_errno(0)
        size = self.lib.proc_listchildpids(pid, array, ctypes.sizeof(array))
        if size < 0 or size == 0 and ctypes.get_errno():
            raise ObserverError("child_inventory_unavailable")
        # Unlike proc_listpids, this wrapper returns a PID count, not bytes.
        if size >= len(array):
            raise ObserverError("process_scope_limit")
        return [p for p in array[:size] if p > 0]


def job_root():
    result = subprocess.run(["/bin/launchctl", "print", f"gui/{os.getuid()}/{JOB}"],
                            capture_output=True, timeout=3)
    text = result.stdout.decode("utf-8", "replace")
    found = re.search(r"\n\s*pid = (\d+)\s*\n", text)
    return int(found[1]) if not result.returncode and "state = running" in text and found else None


def tree(native, root):
    first = native.observation(root["pid"])
    if first is None:
        raise ObserverError("root_counters_unavailable")
    if first["start_abstime"] != root["start_abstime"]:
        raise ObserverError("root_exited_or_reused")
    found = {first["pid"]: first}
    queue = [first]
    while queue:
        parent = queue.pop()
        for pid in native.children(parent["pid"]):
            child = native.observation(pid)
            if child is None or child["ppid"] != parent["pid"] or pid in found:
                continue
            # Refuse a child attached to a different incarnation of its parent.
            current = native.observation(parent["pid"])
            if (current is None or current["start_abstime"] != parent["start_abstime"]
                    or current["ppid"] != parent["ppid"]):
                continue
            found[pid] = child
            queue.append(child)
            if len(found) > MAX_PROCESSES:
                raise ObserverError("process_scope_limit")
    final = native.observation(root["pid"])
    if final is None:
        raise ObserverError("root_counters_unavailable")
    if final["start_abstime"] != root["start_abstime"]:
        raise ObserverError("root_exited_or_reused")
    # Sampling is not atomic. Recheck each ancestry edge and discard an entire
    # branch if its parent disappeared, changed identity or was reparented.
    valid = {final["pid"]: final}
    for pid, old in found.items():
        if pid == root["pid"]:
            continue
        current = native.observation(pid)
        parent = valid.get(old["ppid"])
        if current is None or parent is None:
            continue
        parent_check = native.observation(parent["pid"])
        if (current["start_abstime"] == old["start_abstime"] and current["ppid"] == old["ppid"]
                and parent_check is not None and parent_check["start_abstime"] == parent["start_abstime"]
                and parent_check["ppid"] == parent["ppid"]):
            valid[pid] = current
    return list(valid.values())


def interval_rows(current, previous, numer, denom, logical_cpus):
    rows = []
    for item in current:
        key = (item["pid"], item["start_abstime"])
        old = previous.get(key)
        elapsed = item["monotonic_ns"] - old["monotonic_ns"] if old else 0
        ticks = item["cpu_ticks"] - old["cpu_ticks"] if old else -1
        cpu = (100 * ticks * numer / denom / elapsed
               if old and elapsed > 0 and ticks >= 0 and item["ppid"] == old["ppid"] else None)
        rows.append({"pid": item["pid"], "ppid": item["ppid"],
                     "start_abstime": item["start_abstime"],
                     "cpu_percent_one_core_interval": cpu,
                     "cpu_percent_all_logical_cores_interval": cpu / logical_cpus if cpu is not None else None,
                     "interval_seconds": elapsed / 1e9 if cpu is not None else None,
                     "rss_bytes": item["rss_bytes"],
                     "physical_footprint_bytes": item["physical_footprint_bytes"]})
    return rows


def observe(state_dir, *, max_seconds, interval=10, native=None):
    if not 1 <= max_seconds <= 23400 or not 1 <= interval <= 60:
        raise ObserverError("invalid_observer_budget")
    now = datetime.now(timezone.utc)
    budget = min(max_seconds, max(0, remaining_seconds(now, end=VISION_END) - 10))
    if budget < 1:
        return {"version": VERSION, "status": "skipped", "stop_reason": "outside_observer_window"}
    native = native or Native()
    pid = job_root()
    if pid is None:
        return {"version": VERSION, "status": "skipped", "stop_reason": "no_running_job"}
    root = native.observation(pid) if pid else None
    if root is None:
        return {"version": VERSION, "status": "skipped", "stop_reason": "root_counters_unavailable_or_unowned"}
    folder = state_dir / "process-observations"
    fd = open_private_file(folder / "observer.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"version": VERSION, "status": "skipped", "stop_reason": "observer_already_running"}
        identity = uuid.uuid4().hex
        path = folder / (identity + ".json")
        spool = folder / (identity + ".jsonl")
        source = Path(__file__).resolve()
        receipt = {"version": VERSION, "observer_id": identity, "status": "running",
                   "started_at_utc": now.isoformat(), "cutoff_at_utc": (now + timedelta(seconds=budget)).isoformat(),
                   "root_pid": pid, "root_start_abstime": root["start_abstime"],
                   "sample_file": spool.name, "scope": "exact_owned_overnight_job_and_live_descendants",
                   "attempt_linkage": "none;supplemental_timestamps_only;historical_attempts_unchanged",
                   "cpu_scope": "native_mach_timebase_converted_interval;not_ps_lifetime_cpu",
                   "memory_scope": "per_process;shared_memory_not_summed;not_MLX_allocator",
                   "gpu_scope": "not_collected;use_separate_whole_device_attempt_counters",
                   "mach_timebase": {"numer": native.numer, "denom": native.denom},
                   "logical_cpu_count": os.cpu_count() or 1, "interval_seconds": interval,
                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "network_profile_sha256": hashlib.sha256(source.with_name("network-off.sb").read_bytes()).hexdigest(),
                   "samples": 0, "sample_bytes": 0, "trailing_record_incomplete": False}
        observer_start = native.observation(os.getpid())
        write_json(path, receipt)
        started, deadline = time.monotonic(), time.monotonic() + budget
        digest, previous = hashlib.sha256(), {}
        stop = "observer_deadline"
        out = open_private_file(spool, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            while time.monotonic() < deadline:
                # Recheck wall time after sleep/wake or a clock adjustment.
                if remaining_seconds(datetime.now(timezone.utc), end=VISION_END) <= 10:
                    break
                if job_root() != pid:
                    stop = "job_exited_or_changed"
                    break
                current = tree(native, root)
                row = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                       "elapsed_seconds": round(time.monotonic() - started, 6),
                       "processes": interval_rows(current, previous, native.numer, native.denom,
                                                  receipt["logical_cpu_count"])}
                payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                if receipt["sample_bytes"] + len(payload) > MAX_BYTES:
                    stop = "observer_storage_limit"
                    break
                offset = 0
                receipt["trailing_record_incomplete"] = True
                while offset < len(payload):
                    count = os.write(out, payload[offset:])
                    if count <= 0:
                        raise ObserverError("observer_write_failed")
                    digest.update(payload[offset:offset + count])
                    receipt["sample_bytes"] += count
                    offset += count
                receipt["trailing_record_incomplete"] = False
                receipt["samples"] += 1
                os.fsync(out)
                previous = {(p["pid"], p["start_abstime"]): p for p in current}
                if receipt["samples"] % 5 == 0:
                    write_json(path, receipt)
                time.sleep(max(0, min(interval, deadline - time.monotonic())))
        except ObserverError as error:
            stop = str(error)
        except Exception:
            stop = "observer_failed"
        finally:
            os.close(out)
            observer_end = native.observation(os.getpid())
            own_ticks = (observer_end["cpu_ticks"] - observer_start["cpu_ticks"]
                         if observer_start and observer_end
                         and observer_start["start_abstime"] == observer_end["start_abstime"] else -1)
            receipt.update(status="complete" if stop in {"observer_deadline", "job_exited_or_changed", "root_exited_or_reused"} else "failed",
                           stop_reason=stop, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=round(time.monotonic() - started, 6),
                           sample_sha256=digest.hexdigest(),
                           observer_process_cpu_seconds=own_ticks * native.numer / native.denom / 1e9 if own_ticks >= 0 else None,
                           observer_process_rss_bytes=observer_end["rss_bytes"] if observer_end else None,
                           observer_overhead_scope="observer_process_only;launchctl_helpers_excluded")
            write_json(path, receipt)
        return receipt
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if not args.worker:
            source = Path(__file__).resolve()
            command = ["/usr/bin/sandbox-exec", "-f", str(source.with_name("network-off.sb")),
                       sys.executable, "-B", str(source), *sys.argv[1:], "--worker"]
            result = subprocess.run(command, capture_output=True, timeout=min(args.max_seconds, 23400) + 15)
            if result.returncode:
                raise ObserverError("observer_worker_failed")
            value = json.loads(result.stdout)
        else:
            budget = min(args.max_seconds, max(0, remaining_seconds(datetime.now(timezone.utc), end=VISION_END) - 10))
            if budget > 0:
                # This timer survives loss of the launching chat/process.
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                signal.setitimer(signal.ITIMER_REAL, budget + 8)
            value = observe(args.state_dir, max_seconds=args.max_seconds, interval=args.interval)
        print(json.dumps(value))
        return int(value.get("status") == "failed")
    except Exception as error:
        print(json.dumps({"version": VERSION, "status": "failed",
                          "stop_reason": str(error) if isinstance(error, ObserverError) else "observer_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
