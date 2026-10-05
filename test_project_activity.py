import subprocess
from pathlib import Path

import project_activity
import secure_store


def git(*args, cwd):
    return subprocess.run(["/usr/bin/git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_git_receipts_filter_other_authors_without_reading_file_contents(tmp_path):
    repo = tmp_path / "work"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    git("config", "user.name", "Owner", cwd=repo)
    git("config", "user.email", "owner@example.test", cwd=repo)
    (repo / "work.txt").write_text("private source body")
    git("add", "work.txt", cwd=repo)
    git("commit", "-qm", "local work", cwd=repo)
    owned = git("rev-parse", "HEAD", cwd=repo)
    (repo / "work.txt").write_text("other private body")
    git("add", "work.txt", cwd=repo)
    git("-c", "user.name=Other", "-c", "user.email=other@example.test",
        "commit", "-qm", "other work", cwd=repo)
    rows = project_activity.git_commits(str(repo), 365)
    assert [sha for sha, _ in rows] == [owned]
    assert "private source body" not in repr(rows)
    assert "owner@example.test" not in repr(rows)


def test_repository_discovery_excludes_dependency_trees(tmp_path):
    (tmp_path / "source" / ".git").mkdir(parents=True)
    (tmp_path / "source" / "node_modules" / "nested" / ".git").mkdir(parents=True)
    assert project_activity.discover_repositories(tmp_path) == [tmp_path / "source"]


def make_repo(root):
    root.mkdir()
    git('init', '-q', cwd=root)
    git('config', 'user.name', 'Owner', cwd=root)
    git('config', 'user.email', 'owner@example.test', cwd=root)
    git('commit', '--allow-empty', '-qm', 'fixture', cwd=root)
    return root


def test_home_relative_config_does_not_break_reading(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path/'.gitconfig').write_text('[user]\n\temail = owner@example.test\n')
    repo = make_repo(tmp_path/'work')
    git('config', 'include.path', '~/.gitconfig', cwd=repo)
    assert project_activity.git_commits(str(repo), 365)


def test_bad_ref_keeps_valid_commits_and_reports_incomplete_coverage(tmp_path):
    repo = make_repo(tmp_path/'work')
    broken = repo/'.git/refs/heads/broken'
    broken.write_text('1'*40+'\n')
    diagnostics = {}
    rows = project_activity.git_commits(str(repo), 365, diagnostics=diagnostics)
    assert len(rows) == 1
    assert diagnostics == {'unreadable_ref_tips': 1, 'history_truncated': False}
    assert broken.read_text() == '1'*40+'\n'  # The reader does not repair other repositories.


def test_owner_is_filtered_before_limit_and_committer_matches_are_kept(tmp_path):
    repo = make_repo(tmp_path/'work')
    owned = git('rev-parse', 'HEAD', cwd=repo)
    git('-c', 'user.name=Other', '-c', 'user.email=other@example.test', 'commit',
        '--allow-empty', '--author=Owner <owner@example.test>', '-qm', 'authored work', cwd=repo)
    authored = git('rev-parse', 'HEAD', cwd=repo)
    git('commit', '--allow-empty', '--author=Other <other@example.test>', '-qm', 'committed work', cwd=repo)
    committed = git('rev-parse', 'HEAD', cwd=repo)
    for _ in range(4):
        git('-c', 'user.name=Other', '-c', 'user.email=other@example.test', 'commit',
            '--allow-empty', '-qm', 'other work', cwd=repo)
    diagnostics = {}
    rows = project_activity.git_commits(str(repo), 365, limit=3, diagnostics=diagnostics)
    assert {sha for sha, _ in rows} == {owned, authored, committed}
    assert not diagnostics['history_truncated']
    assert len(project_activity.git_commits(str(repo), 365, limit=2, diagnostics=diagnostics)) == 2
    assert diagnostics['history_truncated']


def test_partial_scan_persists_valid_receipts_idempotently(tmp_path, monkeypatch):
    repo = make_repo(tmp_path/'work')
    (repo/'.git/refs/heads/broken').write_text('1'*40+'\n')
    state = tmp_path/'private-state'
    monkeypatch.setattr(secure_store, 'STATE_DIR', state)
    monkeypatch.setattr(secure_store, 'DB_PATH', state/'ledger.sqlite3')
    monkeypatch.setattr(project_activity, 'STATE_DIR', state)
    monkeypatch.setattr(project_activity, 'STATUS_PATH', state/'git-receipt.json')
    monkeypatch.setattr(project_activity, 'discover_repositories', lambda: [repo, repo/'linked-worktree'])
    monkeypatch.setattr(project_activity, 'resolve_repo', lambda _: (str(repo), str(repo/'.git')))
    result = project_activity.run()
    assert result['status'] == 'partial' and result['errors'] == 0
    assert result['unreadable_ref_tips'] == 1 and result['new_commit_receipts'] == 1
    assert result['git_stores_scanned'] == 1
    assert project_activity.run()['new_commit_receipts'] == 0
    with secure_store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM git_receipts').fetchone()[0] == 1
