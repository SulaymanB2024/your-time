"""Validate public source and apply a separately stored installation-redaction policy."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from urllib.parse import unquote_to_bytes

HOME_PATH = re.compile(rb"(?:file://[^/\s\"'<>`]*|(?<![A-Za-z0-9/]))(?:" + b"/" + b"Users/|" + b"/" + b"home/" + rb")[^/\s\"'<>`]+|[A-Z]:[\\/]" + b"Users" + rb"[\\/][^\\/\s\"'<>`]+", re.IGNORECASE)
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EXAMPLE_DOMAINS = frozenset({b"example.com", b"example.org", b"example.net",
                             b"example.invalid", b"example.test", b"users.noreply.github.com"})
PUBLIC_CONTACTS = frozenset({b"noreply@github.com"})


def load_policy() -> dict:
    path = Path.home() / "Library/Application Support/personal-activity-ledger/publication-policy.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {"version": 1, "replacements": []}
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > 65536):
            raise ValueError("Publication policy must be a bounded private owned file")
        body = stream.read(65537)
    if len(body) > 65536:
        raise ValueError("Publication policy is oversized")
    policy = json.loads(body)
    rules(policy)
    return policy


def rules(policy: dict) -> list[tuple[re.Pattern, bytes]]:
    if not isinstance(policy, dict) or policy.get("version") != 1:
        raise ValueError("Invalid publication policy")
    replacements = policy.get("replacements")
    if not isinstance(replacements, list) or len(replacements) > 100:
        raise ValueError("Invalid publication replacements")
    result = []
    for entry in replacements:
        if (not isinstance(entry, dict) or not isinstance(entry.get("from"), str)
                or not 3 <= len(entry["from"]) <= 1024 or not isinstance(entry.get("to"), str)
                or not 1 <= len(entry["to"]) <= 1024 or entry.get("mode") not in {"literal", "word"}
                or any(char in entry["from"] + entry["to"] for char in "\x00\n\r")):
            raise ValueError("Invalid publication replacement")
        original = entry["from"].encode()
        pattern = re.escape(original)
        flags = 0
        if entry["mode"] == "word":
            pattern = rb"(?<![A-Za-z0-9])" + pattern + rb"(?![A-Za-z0-9])"
            flags = re.IGNORECASE
        result.append((re.compile(pattern, flags), entry["to"].encode()))
    return result


def public_text(body: bytes, policy: dict) -> bytes:
    """Redact explicit private identifiers, then refuse remaining personal paths/contact data."""
    compiled = rules(policy)
    for pattern, replacement in compiled:
        body = pattern.sub(lambda _: replacement, body)
        if len(body) > 2 * 1024**2:
            raise ValueError("Redacted public source is oversized")
    normalized = unquote_to_bytes(body.replace(b"\\/", b"/"))
    if HOME_PATH.search(normalized):
        raise ValueError("Public source contains a concrete personal home path")
    if any(email.rsplit(b"@", 1)[1].lower() not in EXAMPLE_DOMAINS
           and email.lower() not in PUBLIC_CONTACTS for email in EMAIL.findall(normalized)):
        raise ValueError("Public source contains non-example contact information")
    if any(pattern.search(normalized) for pattern, _ in compiled):
        raise ValueError("Private installation identifier remains in public source")
    return body
