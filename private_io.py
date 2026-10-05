"""Private POSIX file operations shared by collectors and analysis jobs.

Files are owned regular files with a single link. Directory traversal and file
opens reject symlinks; writes use an exclusive random temporary file and rename
relative to an open directory descriptor. This does not isolate a compromised
process already running as the same user.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import stat
from pathlib import Path


def _check_file(info: os.stat_result, *, private: bool = True, snapshot: bool = False) -> None:
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink not in ({0, 1} if snapshot else {1})
            or (private and info.st_mode & 0o077)):
        raise RuntimeError("Expected a private owned regular file with one link")


def _directory_fd(path: Path) -> int:
    """Open/create each directory without following links; secure only the leaf."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError("Private directory must be an absolute non-root path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass  # A concurrent creator still has to pass the open below.
                child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise RuntimeError("Private directory must be owned by this user")
        os.fchmod(fd, 0o700)
        return fd
    except BaseException:
        os.close(fd)
        raise


def prepare_directory(path: Path) -> None:
    os.close(_directory_fd(path))


def _existing_file(directory: int, name: str, *, private: bool = True) -> None:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise OSError(errno.ELOOP, "Private file cannot be a symbolic link")
    # macOS can return nlink=0 for a directory-entry stat racing another atomic
    # replacement. That obsolete regular inode cannot be a hard-link escape.
    # Open descriptors still require exactly one link below.
    _check_file(info, private=private, snapshot=True)


def validate_file(path: Path) -> None:
    """Validate an existing private file, allowing an absent SQLite sidecar."""
    directory = _directory_fd(path.parent)
    try:
        _existing_file(directory, path.name)
    finally:
        os.close(directory)


def open_private_file(path: Path, flags: int = os.O_RDWR | os.O_CREAT) -> int:
    """Return an owned private descriptor; validate before any truncation."""
    directory = _directory_fd(path.parent)
    fd = None
    try:
        _existing_file(directory, path.name)
        fd = os.open(path.name, (flags & ~os.O_TRUNC) | os.O_NOFOLLOW
                     | os.O_CLOEXEC | os.O_NONBLOCK, 0o600, dir_fd=directory)
        _check_file(os.fstat(fd))
        if flags & os.O_TRUNC:
            os.ftruncate(fd, 0)
        return fd
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise
    finally:
        os.close(directory)


def atomic_write(path: Path, payload: bytes) -> None:
    """Replace one private file atomically, cleaning up after failed writes."""
    directory = _directory_fd(path.parent)
    temporary = ".private-" + secrets.token_hex(16) + ".tmp"
    created = False
    try:
        # An old nonprivate regular output can be replaced with a private file;
        # linked or foreign-owned targets must never be silently replaced.
        _existing_file(directory, path.name, private=False)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        created = True
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        _existing_file(directory, path.name, private=False)
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        created = False
        try:
            os.fsync(directory)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        if created:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
