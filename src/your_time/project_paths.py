"""Separate work-file metadata from generated and runtime churn."""

from pathlib import PurePath

from project_activity import SKIP_DIRS

GENERATED_DIRS = SKIP_DIRS | frozenset({
    "out", "static-pages", "static-catalog", "logs", "runtime", "state",
    "site-packages", "test-results", "playwright-report", "target",
})
RUNTIME_SUFFIXES = (".log", ".sqlite", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm",
                    ".db", ".db-wal", ".db-shm", ".tmp", ".swp", "~")


def is_work_path(relative: str | PurePath) -> bool:
    path = PurePath(relative)
    return (bool(path.parts) and not path.is_absolute()
            and not any(part in GENERATED_DIRS or part.startswith(".") or part == ".."
                        for part in path.parts)
            and not path.name.casefold().endswith(RUNTIME_SUFFIXES))
