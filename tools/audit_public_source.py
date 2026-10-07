"""Scan public Git ancestry; report only categories/counts/locations, never matched text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter

from repo_layout import REPO_ROOT, SOURCE_ROOT

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

from publication_privacy import (
    EMAIL,
    EXAMPLE_DOMAINS,
    HOME_PATH,
    PUBLIC_CONTACTS,
    load_policy,
    normalized_text,
    rules,
)
from publish_source import SECRETS


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(REPO_ROOT), *args], stderr=subprocess.DEVNULL, timeout=120)


def audit(refs: list[str]) -> dict:
    if not refs or len(refs) > 30 or any(ref.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_./-]{1,200}', ref) for ref in refs):
        raise ValueError('Invalid audit refs')
    commits = git('rev-list', *refs).decode().splitlines()
    objects = git('rev-list', '--objects', *refs).decode().splitlines()
    names = {line.split(' ', 1)[0]: line.split(' ', 1)[1] if ' ' in line else None for line in objects}
    identifiers = rules(load_policy())
    totals = Counter()
    findings = []
    blobs = 0
    metadata_findings = []
    filename_findings = []

    def classify(line: bytes) -> dict:
        line = normalized_text(line)
        return {'personal_home_path': bool(HOME_PATH.search(line)),
                'known_private_identifier': any(pattern.search(line) for pattern, _ in identifiers),
                'credential_pattern': bool(SECRETS.search(line)),
                'non_example_contact': any(email.rsplit(b'@', 1)[1].lower() not in EXAMPLE_DOMAINS
                                           and email.lower() not in PUBLIC_CONTACTS for email in EMAIL.findall(line))}

    def safe_path(path: str | None) -> str:
        # Contact/home-path matches are reported by hash rather than echoed.
        encoded = (path or '<unnamed>').encode()
        checks = classify(encoded)
        if checks['personal_home_path'] or checks['non_example_contact'] or checks['credential_pattern']:
            return 'path-sha256:' + hashlib.sha256(encoded).hexdigest()
        for pattern, replacement in identifiers:
            encoded = pattern.sub(lambda _: replacement, encoded)
        if classify(encoded)['known_private_identifier']:
            return 'path-sha256:' + hashlib.sha256(encoded).hexdigest()
        return encoded.decode(errors='replace')

    # A blob may occur under several filenames; inspect every reachable path.
    seen_paths = set()
    for commit in commits:
        for entry in git('ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
            if not entry:
                continue
            metadata, raw_path = entry.split(b'\t', 1)
            oid = metadata.split()[2].decode()
            identity = (oid, raw_path)
            if identity in seen_paths:
                continue
            seen_paths.add(identity)
            if len(seen_paths) > 200000:
                raise ValueError('Public tree history exceeds the bounded audit')
            categories = [name for name, matched in classify(raw_path).items() if matched]
            if categories:
                filename_findings.append({'object': oid, 'path': safe_path(raw_path.decode()), 'categories': categories})
    process = subprocess.Popen(['git', '-C', str(REPO_ROOT), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for oid, path in names.items():
            process.stdin.write((oid + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3:
                raise ValueError('Git object unavailable')
            size = int(header[2])
            if size > 4 * 1024**2:
                raise ValueError('Oversized public Git object requires separate review')
            body = process.stdout.read(size)
            if len(body) != size or process.stdout.read(1) != b'\n':
                raise ValueError('Incomplete Git object')
            if header[1] not in {b'blob', b'commit'}:
                continue
            is_blob = header[1] == b'blob'
            if is_blob:
                blobs += 1
            categories = Counter()
            locations = {}
            for number, line in enumerate(body.splitlines(), 1):
                checks = classify(line)
                for category, matched in checks.items():
                    if matched:
                        categories[category] += 1
                        locations.setdefault(category, []).append(number)
            if b'\0' in body:
                categories['binary_blob'] += 1
            if categories:
                if is_blob:
                    totals.update(categories)
                # Paths may themselves identify the installation; normalize them.
                item = {'object': oid, 'path': safe_path(path),
                        'categories': dict(categories), 'lines': locations}
                (findings if is_blob else metadata_findings).append(item)
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait(timeout=10)
    return {'version': 'public_source_privacy_audit_v2', 'refs': refs,
            'reachable_commits': len(set(commits)), 'unique_blobs': blobs,
            'matched_blobs': len(findings), 'category_blob_lines': dict(totals),
            'matched_commit_metadata': len(metadata_findings), 'commit_metadata_findings': metadata_findings,
            'matched_filenames': len(filename_findings), 'filename_findings': filename_findings,
            'findings': findings,
            'limits': ['Git refs and blobs only; hosted caches and external clones are outside this scan',
                       'pattern results require review; absence of matches is not proof of absence']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(audit(args.ref or ['refs/heads/codex/github-source']), sort_keys=True))
