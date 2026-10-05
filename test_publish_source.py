import subprocess

import pytest

from publish_source import (REMOTE, SOURCE_REF, allowed, git, source_commit,
                            source_entries, source_tree, validate_push)


@pytest.mark.parametrize('path', [
    '.env', 'activity.sqlite3', 'model.gguf', 'capture.png', 'analyses/day.json',
    'vision-latest-receipt.json', 'PERFORMANCE_REVIEW_2026-10-04.md',
    'VISION_QUALITY_TEST.md', 'unknown.json', '../secret.py', '/root.py',
])
def test_runtime_data_and_private_notes_are_excluded(path):
    assert not allowed(path)


def fixture_repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    git(tmp_path, 'config', 'user.name', 'Fixture')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    return tmp_path


def commit_files(repo):
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'fixture')


def test_clean_snapshot_has_no_private_ancestry_and_preserves_checkout(tmp_path):
    repo = fixture_repo(tmp_path)
    (repo/'private-notes.md').write_text('PRIVATE HISTORY')
    commit_files(repo)
    (repo/'README.md').write_text('Source documentation')
    (repo/'worker.py').write_text('print(1)')
    (repo/'activity.sqlite3').write_bytes(b'PRIVATE DATABASE')
    commit_files(repo)
    original = git(repo, 'rev-parse', 'HEAD')
    index = (repo/'.git/index').read_bytes()
    tree = source_tree(repo, source_entries(repo))
    clean = source_commit(repo, tree)
    assert git(repo, 'rev-list', '--count', clean).strip() == b'1'
    assert set(git(repo, 'ls-tree', '-r', '--name-only', clean).decode().splitlines()) == {'README.md', 'worker.py'}
    assert git(repo, 'rev-parse', 'HEAD') == original
    assert (repo/'.git/index').read_bytes() == index
    assert git(repo, 'status', '--porcelain') == b''
    # The next export has only the previous source commit as its parent.
    (repo/'worker.py').write_text('print(2)')
    commit_files(repo)
    second = source_commit(repo, source_tree(repo, source_entries(repo)), clean)
    assert git(repo, 'rev-list', '--count', second).strip() == b'2'
    assert git(repo, 'rev-parse', second + '^').strip() == clean.encode()


def test_selected_symlink_is_rejected(tmp_path):
    repo = fixture_repo(tmp_path)
    (repo/'README.md').symlink_to('/private/source')
    commit_files(repo)
    with pytest.raises(RuntimeError, match='rejects links'):
        source_entries(repo)


def test_credentials_in_selected_source_block_publication(tmp_path):
    repo = fixture_repo(tmp_path)
    token = 'gh' + 'p_' + 'x'*40
    (repo/'worker.py').write_text('token = ' + repr(token))
    commit_files(repo)
    with pytest.raises(RuntimeError, match='credential'):
        source_entries(repo)


def test_push_guard_rejects_private_ancestry_and_extra_refs(tmp_path):
    repo = fixture_repo(tmp_path)
    (repo/'README.md').write_text('Source')
    (repo/'private.md').write_text('PRIVATE HISTORY')
    commit_files(repo)
    working = git(repo, 'rev-parse', 'HEAD').decode().strip()
    clean = source_commit(repo, source_tree(repo, source_entries(repo)))
    zero = '0'*40
    record = lambda commit, destination='refs/heads/main', old=zero: f'HEAD {commit} {destination} {old}\n'
    validate_push(repo, REMOTE, record(clean))
    with pytest.raises(RuntimeError, match='excluded'):
        validate_push(repo, REMOTE, record(working))
    with pytest.raises(RuntimeError, match='only a source commit'):
        validate_push(repo, REMOTE, record(clean, 'refs/heads/private'))
    with pytest.raises(RuntimeError, match='exactly one'):
        validate_push(repo, REMOTE, record(clean)*2)
    git(repo, 'update-ref', SOURCE_REF, clean)
    second = source_commit(repo, source_tree(repo, source_entries(repo)), clean)
    validate_push(repo, REMOTE, record(second, old=clean))
    with pytest.raises(RuntimeError, match='descended solely'):
        validate_push(repo, REMOTE, record(working, old=clean))
