import subprocess
from pathlib import Path

import project_activity


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
