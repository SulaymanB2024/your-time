import json
import subprocess

import audit_public_source as audit_module
from publish_source import git


def repo_fixture(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    git(tmp_path, 'config', 'user.name', 'Fixture')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    monkeypatch.setattr(audit_module, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(audit_module, 'load_policy', lambda: {'version': 1, 'replacements': [
        {'from': 'private-human', 'to': 'yourtime', 'mode': 'word'}]})
    return tmp_path


def commit(repo):
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'fixture')


def test_audit_covers_percent_encoded_bodies_and_commit_metadata(tmp_path, monkeypatch):
    repo = repo_fixture(tmp_path, monkeypatch)
    home = '/'.join(['', 'Users', 'private-human', 'file'])
    token = 'gh' + 'p_' + 'x'*40
    encoded = ''.join('%' + format(ord(char), '02X') for char in token)
    (repo/'README.md').write_text(home.replace('/', '%2F') + '\n' + encoded)
    contact = 'person' + '@' + 'personal-mail.net'
    git(repo, 'config', 'user.email', contact)
    commit(repo)
    result = audit_module.audit(['HEAD'])
    assert result['category_blob_lines']['personal_home_path'] == 1
    assert result['category_blob_lines']['credential_pattern'] == 1
    assert result['matched_commit_metadata'] == 1
    serialized = json.dumps(result)
    assert token not in serialized and encoded not in serialized
    assert contact not in serialized and home not in serialized


def test_audit_checks_every_historical_filename_even_for_reused_blobs(tmp_path, monkeypatch):
    repo = repo_fixture(tmp_path, monkeypatch)
    (repo/'README.md').write_text('Same harmless source')
    commit(repo)
    (repo/'private-human-guide.md').write_text('Same harmless source')
    contact_filename = 'person' + '%' + '40' + 'personal-mail.net'
    (repo/contact_filename).write_text('Same harmless source')
    commit(repo)
    (repo/'private-human-guide.md').unlink()
    (repo/contact_filename).unlink()
    commit(repo)
    result = audit_module.audit(['HEAD'])
    assert result['unique_blobs'] == 1
    assert result['matched_filenames'] == 2
    assert result['matched_blobs'] == 0
    serialized = json.dumps(result)
    assert 'private-human' not in serialized and 'personal-mail.net' not in serialized
    assert any(item['path'].startswith('path-sha256:') for item in result['filename_findings'])


def test_safe_example_contacts_and_public_attribution_do_not_flag(tmp_path, monkeypatch):
    repo = repo_fixture(tmp_path, monkeypatch)
    (repo/'README.md').write_text('fixture@example.invalid noreply@github.com')
    commit(repo)
    result = audit_module.audit(['HEAD'])
    assert result['matched_blobs'] == result['matched_commit_metadata'] == result['matched_filenames'] == 0
