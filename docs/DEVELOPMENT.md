# Development

## Supported environment

macOS on Apple Silicon, Python 3.11, Node.js and the macOS `zsh` and
`sandbox-exec` utilities. The dependency lock pins the Python environment.
This repository represents an installed configuration, not a portable installer.
Do not run collection wrappers on a different account without reviewing paths,
scope and permissions first.

```sh
uv sync --locked
make check
```

`make check` runs Ruff, parses Python/JSON/plist files, checks shell and
JavaScript syntax, and runs synthetic tests. It does not open the activity
database, take screenshots, run a model, change privacy settings or load jobs.
Native APIs are mocked in capture tests; the child-lock test runs a short
synthetic process under the network-denying sandbox.

For a focused change:

```sh
.venv/bin/pytest -q tests/test_private_io.py tests/test_secure_store.py
make lint
```

For this installation, `make setup` performs a separate read-only check of
collector freshness, private permissions, signatures, SQLite integrity and
network denial. `chronicle_audit.py` checks accounting bounds; it cannot grade
whether a caption correctly describes an image.

`make audit` is an optional dependency-advisory query. It sends package names
and versions to public advisory services, with no activity data. Its result is
limited to published advisories and this environment, not separately packaged
reader binaries. Audits do not automatically upgrade dependencies.

## Change rules

1. Preserve observation timestamps, device separation and unknown intervals.
2. Use `private_io` for runtime writes and lock descriptors. Never use a
   predictable truncating temporary path or weaken privacy checks to recover
   from a failure. Keep private content out of command arguments and error logs.
3. Models receive bounded untrusted evidence and have no tools or network.
   Keep supported context separate from verified accomplishments.
4. Add regression tests for a real behavior or boundary, with synthetic data.
5. Signed application changes require a separately reviewed build and permission
   readback. Editing source does not update a pinned installed app.
6. Check an installed service after a relevant change. Restart only the exact
   affected service; let queued metadata flush before restarting its watcher.
7. Review source diffs and register new root modules/documents deliberately in
   `publish_source.py`. Never push the private historical branch directly.

## Public delivery

Tests live under `tests/`, engineering documentation under `docs/`, and source
checks under `tools/`. See [architecture](ARCHITECTURE.md), the
[security boundary](../SECURITY.md), and [source publication](../GITHUB.md).
The hosted source-only test workflow is described in [CI](CI.md).
