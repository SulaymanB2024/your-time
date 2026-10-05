"""Preview or explicitly push a source-only snapshot; never push private history."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

REPOSITORY = 'SulaymanB2024/your-time'
REMOTE = f'https://github.com/{REPOSITORY}.git'
SOURCE_REF = 'refs/heads/codex/github-source'
CI_FILES = {'.github/workflows/checks.yml'}
DOCS = {'.gitignore', 'README.md', 'SECURITY.md', 'GITHUB.md', 'uv.lock', 'Makefile',
        'docs/ARCHITECTURE.md', 'docs/DEVELOPMENT.md', 'docs/CI.md'}
ROOT_SOURCE_FILES = {
    'behavior_analysis.py',
    'browser_bridge.py',
    'build_calendar_reader.zsh',
    'build_window_reader.zsh',
    'calendar_import.py',
    'capture_control.zsh',
    'chronicle_audit.py',
    'chronicle_rollup.py',
    'daily_analysis.py',
    'daily_export.py',
    'daily_focus.py',
    'dashboard_explore.py',
    'dashboard_fixture.py',
    'dashboard_styles.py',
    'dashboard_timeline.py',
    'dashboard_ui.py',
    'data_quality.py',
    'download_quality_model.py',
    'download_vision_model.py',
    'import_legacy_mac.py',
    'index_features.py',
    'index_screenshots.py',
    'install_browser_host.py',
    'local_dashboard.py',
    'local_synthesis.py',
    'mac_activity.py',
    'model_execution.py',
    'network-off.sb',
    'open_dashboard.zsh',
    'overnight_schedule.py',
    'overnight_text.py',
    'overnight_vision.zsh',
    'phone_import.py',
    'phone_quality.py',
    'private_io.py',
    'project_activity.py',
    'project_file_watch.py',
    'project_paths.py',
    'publish_source.py',
    'pyproject.toml',
    'resume_activity_once.zsh',
    'screen_context_tagging.py',
    'screen_similarity.py',
    'screen_storage_guard.py',
    'secure_analyze.zsh',
    'secure_browser_bridge.zsh',
    'secure_calendar.zsh',
    'secure_capture.py',
    'secure_dashboard.zsh',
    'secure_features.zsh',
    'secure_index.zsh',
    'secure_mac.zsh',
    'secure_phone.zsh',
    'secure_project_file_watch.zsh',
    'secure_project_git.zsh',
    'secure_record.zsh',
    'secure_store.py',
    'secure_text_analysis.zsh',
    'secure_vision_fallback.zsh',
    'secure_window_reader.zsh',
    'setup_check.py',
    'task_corrections.py',
    'topic_allocation.py',
    'vision_batch.py',
    'vision_capacity.py',
    'vision_fallback.py',
    'vision_quality_eval.py',
    'window_topic_tagging.py',
    'workstream_tagging.py',
}
JSON_FILES = {'vision_model_manifest.json', 'vision_quality_model_manifest.json',
              'vision_peak_model_manifest.json', 'browser_extension/manifest.json',
              'native/calendar-scope.template.json'}
SECRETS = re.compile(
    rb'(?:ghp_|gho_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{20,}'
    rb'|(?:AKIA|ASIA)[A-Z0-9]{16}'
    rb'|AIza[A-Za-z0-9_-]{35}|(?:glpat-|hf_)[A-Za-z0-9_-]{20,}'
    rb'|xox[baprs]-[A-Za-z0-9-]{20,}'
    rb'|-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----'
    rb'|data:image/[^;\s]+;base64,[A-Za-z0-9+/]{100,}')


def git(repo: Path, *args: str, data: bytes | None = None, env: dict | None = None) -> bytes:
    result = subprocess.run(['git', '-C', str(repo), *args], input=data, capture_output=True,
                            timeout=120, env=env)
    if result.returncode:
        # Git errors may contain paths or remote credentials; do not echo them.
        raise RuntimeError('Git operation failed: ' + args[0])
    return result.stdout


def allowed(path: str) -> bool:
    item = PurePosixPath(path)
    if item.is_absolute() or '..' in item.parts:
        return False
    if path in DOCS or path in JSON_FILES or path in CI_FILES:
        return True
    if len(item.parts) == 1:
        return path in ROOT_SOURCE_FILES
    return (len(item.parts) == 2 and
            ((item.parts[0] == 'tests' and item.name.startswith('test_') and item.suffix == '.py') or
             (path == 'tests/conftest.py') or
             (path == 'tools/check.py') or
             (item.parts[0] == 'launchagents' and item.suffix == '.plist') or
             (item.parts[0] == 'native' and item.suffix in {'.m', '.plist'}) or
             (item.parts[0] == 'browser_extension' and item.suffix in {'.js', '.css', '.html'})))


def source_entries(repo: Path, revision: str = 'HEAD') -> list[bytes]:
    entries = []
    for entry in git(repo, 'ls-tree', '-rz', '--full-tree', revision).split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        path = raw_path.decode('utf-8')
        if not allowed(path):
            continue
        mode, kind, oid = metadata.split()
        if mode not in (b'100644', b'100755') or kind != b'blob':
            raise RuntimeError('Source export rejects links and non-file entries')
        body = git(repo, 'cat-file', 'blob', oid.decode())
        if len(body) > 2 * 1024**2 or b'\0' in body or SECRETS.search(body):
            raise RuntimeError('Source export blocked by binary/credential/image check')
        body.decode('utf-8')
        entries.append(mode + b' ' + oid + b'\t' + raw_path)
    if not entries:
        raise RuntimeError('Empty source export')
    return entries


def source_tree(repo: Path, entries: list[bytes]) -> str:
    # A separate temporary index leaves the live checkout and its index alone.
    with tempfile.TemporaryDirectory(prefix='your-time-source-') as directory:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / 'index'))
        git(repo, 'read-tree', '--empty', env=environment)
        git(repo, 'update-index', '-z', '--index-info', data=b'\0'.join(entries) + b'\0', env=environment)
        return git(repo, 'write-tree', env=environment).decode().strip()


def source_commit(repo: Path, tree: str, parent: str | None = None) -> str:
    # Only prior published source is a parent, never the local working branch.
    environment = dict(os.environ,
        GIT_AUTHOR_NAME='SulaymanB2024', GIT_COMMITTER_NAME='SulaymanB2024',
        GIT_AUTHOR_EMAIL='191058572+SulaymanB2024@users.noreply.github.com',
        GIT_COMMITTER_EMAIL='191058572+SulaymanB2024@users.noreply.github.com')
    args = ['commit-tree', tree, '-m', 'chore(source): publish verified source snapshot']
    if parent:
        args += ['-p', parent]
    return git(repo, *args, env=environment).decode().strip()


def validate_push(repo: Path, remote: str, records: str) -> None:
    """Local pre-push guard: reject private history, extra refs and deletions."""
    if remote != REMOTE:
        raise RuntimeError('Push blocked: unexpected repository')
    previous = git(repo, 'for-each-ref', '--format=%(objectname)', SOURCE_REF).decode().strip()
    lines = records.splitlines()
    if len(lines) != 1:
        raise RuntimeError('Push blocked: exactly one source ref is required')
    fields = lines[0].split()
    if len(fields) != 4:
        raise RuntimeError('Push blocked: invalid ref update')
    _, commit, destination, old = fields
    if destination != 'refs/heads/main' or not re.fullmatch(r'[0-9a-f]{40}', commit) or set(commit) == {'0'}:
        raise RuntimeError('Push blocked: only a source commit to main is allowed')
    parents = git(repo, 'rev-list', '--parents', '-n', '1', commit).decode().split()[1:]
    expected = [previous] if previous else []
    if parents != expected or old != (previous or '0'*40):
        raise RuntimeError('Push blocked: commit is not descended solely from published source')
    exported = source_tree(repo, source_entries(repo, commit))
    if exported != git(repo, 'rev-parse', commit + '^{tree}').decode().strip():
        raise RuntimeError('Push blocked: commit contains excluded data or notes')


def verify_repository() -> None:
    result = subprocess.run(['gh', 'repo', 'view', REPOSITORY, '--json', 'isPrivate,nameWithOwner'],
                            capture_output=True, text=True, timeout=30)
    try:
        metadata = json.loads(result.stdout) if not result.returncode else None
    except ValueError:
        metadata = None
    if metadata != {'isPrivate': False, 'nameWithOwner': REPOSITORY}:
        raise RuntimeError('Approved public repository identity could not be verified')


def push_source(repo: Path, tree: str) -> str:
    origin = git(repo, 'remote', 'get-url', 'origin').decode().strip()
    if origin != REMOTE:
        raise RuntimeError('Origin is not the approved Your Time repository')
    verify_repository()
    heads = git(repo, 'ls-remote', '--heads', 'origin', 'refs/heads/main').split()
    parent = heads[0].decode() if heads else None
    local = git(repo, 'for-each-ref', '--format=%(objectname)', SOURCE_REF).decode().strip()
    if parent:
        # Refuse to attach this export to an unknown remote history.
        if parent != local:
            raise RuntimeError('Published history changed; reconcile before pushing')
        if git(repo, 'rev-parse', parent + '^{tree}').decode().strip() == tree:
            return parent
    commit = source_commit(repo, tree, parent)
    # Explicit destination; no force, branches, tags, or private ancestry.
    git(repo, 'push', 'origin', commit + ':refs/heads/main')
    confirmed = git(repo, 'ls-remote', '--heads', 'origin', 'refs/heads/main').split()
    if not confirmed or confirmed[0].decode() != commit:
        raise RuntimeError('Remote source commit was not confirmed')
    git(repo, 'update-ref', SOURCE_REF, commit)
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', action='store_true', help='Explicitly publish to the approved public source repository')
    parser.add_argument('--check-push', metavar='REMOTE_URL', help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    if args.check_push:
        validate_push(repo, args.check_push, sys.stdin.read())
        return
    if git(repo, 'status', '--porcelain', '--untracked-files=normal').strip():
        raise RuntimeError('Commit or preserve local changes before publishing')
    entries = source_entries(repo)
    report = {'repository': REPOSITORY, 'source_files': len(entries),
              'excluded_history': True, 'pushed': False}
    if args.push:
        report.update(commit=push_source(repo, source_tree(repo, entries)), pushed=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(str(error))
