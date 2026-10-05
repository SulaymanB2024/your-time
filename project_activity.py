"""Collect local Git artifact receipts from explicitly approved project roots."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from private_io import open_private_file, write_json
from secure_store import STATE_DIR, connect, prepare_private_dir

APPROVED_ROOTS = (Path.home() / "Projects", Path.home() / "Projects/CodexWork")
SCAN_ROOT = Path.home() / "Projects"  # CodexWork is inside this root.
STORAGE_NAV = Path.home() / ".local/bin/storage-nav"
STATUS_PATH = STATE_DIR / "project-git-latest-receipt.json"
LOCK_PATH = STATE_DIR / "project-git-scan.lock"
SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", "dist", "build", ".next",
                       ".cache", "vendor", "coverage", "__pycache__", "data"})
MAX_DEPTH = 4


def git_read(path: str, *args: str, input_text: str | None = None, timeout: int = 30):
    return subprocess.run(["/usr/bin/git", "-C", path, "--no-pager", *args],
                          input=input_text, capture_output=True, text=True, timeout=timeout,
                          check=False, env={"HOME": str(Path.home()),
                                            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                                            "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
                                            "GIT_NO_LAZY_FETCH": "1"})


def discover_repositories(root: Path = SCAN_ROOT) -> list[Path]:
    found = []
    for dirname, dirs, files in os.walk(root, followlinks=False):
        here = Path(dirname)
        depth = len(here.relative_to(root).parts)
        if ".git" in dirs or ".git" in files:
            found.append(here)
        dirs[:] = [item for item in dirs if item not in SKIP_DIRS
                   and not (here / item).is_symlink() and depth < MAX_DEPTH]
    return sorted(set(found))


def resolve_repo(path: Path) -> tuple[str, str] | None:
    result = subprocess.run([str(STORAGE_NAV), "resolve", str(path)],
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        return None
    fields = dict(line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line)
    top, common = fields.get("git_top_level"), fields.get("git_common_dir")
    if not top or top == "-" or not common or common == "-":
        return None
    resolved = Path(top).resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in APPROVED_ROOTS):
        return None
    return str(resolved), common


def local_git_emails(path: str) -> set[str]:
    emails = set()
    for scope in ((), ('--global',)):
        result = git_read(path, 'config', *scope, '--get', 'user.email', timeout=5)
        if result.returncode == 0 and "@" in result.stdout:
            emails.add(result.stdout.strip().casefold())
    return emails


def git_commits(path: str, since_days: int, limit: int = 500, *, diagnostics: dict | None = None) -> list[tuple[str, str]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(unreadable_ref_tips=0, history_truncated=False)
    emails = local_git_emails(path)
    if not emails:
        return []
    refs = git_read(path, 'for-each-ref', '--format=%(objectname)')
    if refs.returncode:
        raise RuntimeError("git_log_failed")
    tips = sorted({line for line in refs.stdout.splitlines()
                   if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', line)})
    if tips:
        objects = git_read(path, 'cat-file', '--batch-check=%(objectname) %(objecttype)',
                           input_text='\n'.join(tips) + '\n')
        if objects.returncode:
            raise RuntimeError("git_log_failed")
        diagnostics['unreadable_ref_tips'] = sum(line.endswith(' missing')
                                               for line in objects.stdout.splitlines())
    # Filter before bounding the history. Busy collaborators must not crowd
    # out the owner's commits. Separate queries express author OR committer.
    pattern = '<(' + '|'.join(re.escape(email) for email in sorted(emails)) + ')>'
    rows = {}
    for field in ('author', 'committer'):
        result = git_read(path, 'log', '--ignore-missing', '--all', '--no-show-signature',
                          '--no-color', '--extended-regexp', '--regexp-ignore-case',
                          f'--{field}={pattern}', f'--since={since_days} days ago',
                          f'-n{limit + 1}', '--format=%H%x09%cI%x09%ae%x09%ce')
        if result.returncode:
            head = git_read(path, 'rev-parse', '--verify', 'HEAD', timeout=5)
            if head.returncode:
                return []
            raise RuntimeError("git_log_failed")
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 4 or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', parts[0]):
                continue
            if not (parts[2].casefold() in emails or parts[3].casefold() in emails):
                continue
            try:
                committed = datetime.fromisoformat(parts[1]).astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
            rows[parts[0]] = committed
    ordered = sorted(rows.items(), key=lambda row: (row[1], row[0]), reverse=True)
    diagnostics['history_truncated'] = len(ordered) > limit
    return ordered[:limit]


def write_receipt(value: dict) -> None:
    write_json(STATUS_PATH, {"checked_at_utc": datetime.now(timezone.utc).isoformat(), **value})


def run(*, since_days: int = 365, limit_repos: int | None = None) -> dict:
    if not 1 <= since_days <= 3650:
        raise ValueError("since_days must be 1-3650")
    if limit_repos is not None and limit_repos < 1:
        raise ValueError("limit_repos must be positive")
    candidates = discover_repositories()
    if limit_repos is not None:
        candidates = candidates[:limit_repos]
    started = time.monotonic()
    resolved = scanned = inserted = errors = 0
    error_counts = Counter()
    seen_common = set()
    unreadable_refs = truncated_stores = 0
    for candidate in candidates:
        try:
            identity = resolve_repo(candidate)
            if not identity:
                continue
            repo_path, common_dir = identity
            resolved += 1
            # Shared branches and linked worktrees may point to one Git store.
            common_key = hashlib.sha256(common_dir.encode()).hexdigest()[:20]
            if common_key in seen_common:
                continue
            seen_common.add(common_key)
            diagnostics = {}
            commits = git_commits(repo_path, since_days, diagnostics=diagnostics)
            unreadable_refs += diagnostics['unreadable_ref_tips']
            truncated_stores += int(diagnostics['history_truncated'])
            scanned += 1
            now = datetime.now(timezone.utc).isoformat()
            with connect() as database:
                for sha, committed in commits:
                    identity = hashlib.sha256((common_key + sha).encode()).hexdigest()
                    cursor = database.execute(
                        "INSERT OR IGNORE INTO git_receipts VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (identity, common_key, repo_path, sha, committed, now, "local_commit"),
                    )
                    inserted += cursor.rowcount
        except subprocess.TimeoutExpired:
            errors += 1
            error_counts["timeout"] += 1
        except RuntimeError:
            errors += 1
            error_counts["git_log_failed"] += 1
        except sqlite3.Error:
            errors += 1
            error_counts["database_write_failed"] += 1
        except (OSError, subprocess.SubprocessError, ValueError):
            errors += 1
            error_counts["other"] += 1
    result = {"status": "partial" if errors or unreadable_refs or truncated_stores else "complete",
              "candidate_repositories": len(candidates), "resolved": resolved,
              "git_stores_scanned": scanned, "new_commit_receipts": inserted,
              "errors": errors, "elapsed_seconds": round(time.monotonic() - started, 2),
              "error_counts": dict(error_counts),
              "unreadable_ref_tips": unreadable_refs,
              "history_truncated_stores": truncated_stores,
              "since_days": since_days,
              "interpretation": "Local commits are artifacts, not proof of deployment or acceptance."}
    write_receipt(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since-days", type=int, default=365)
    parser.add_argument("--limit-repos", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    prepare_private_dir()
    fd = open_private_file(LOCK_PATH)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "another_git_scan_is_running"}))
            return
        print(json.dumps(run(since_days=args.since_days, limit_repos=args.limit_repos), sort_keys=True))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
