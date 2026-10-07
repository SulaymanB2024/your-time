# Development

## Supported environment

macOS 14 or later on Apple Silicon, Python 3.11, Node.js and the macOS `zsh` and
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
The test configuration selects a temporary home before importing runtime
modules and rejects Python file/SQLite access to the user's real `Library`.
Database and image fixtures remain synthetic even when running tests on an
installed Mac. This guard prevents accidental access; it is not an OS sandbox
for arbitrary native calls or subprocesses.
The synthetic feature-print integration runs Apple's real Vision framework on
the CPU. macOS 14 uses the older CPU-only request flag; newer systems select
supported CPU compute stages. A cold hosted macOS 14 run required the older
flag to initialize Vision. The integration checks repeatability, stored vector
format, and a changed-image distance; it remains required in hosted CI.

For a focused change:

```sh
.venv/bin/pytest -q tests/test_private_io.py tests/test_secure_store.py
make lint
```

For this installation, `make setup` performs a separate read-only check of
collector freshness, private permissions, signatures, SQLite integrity and
network denial. `chronicle_audit.py` checks accounting bounds; it cannot grade
whether a caption correctly describes an image.

### Dashboard performance

Edit behavior in `dashboard_metrics.py`, rendering in `dashboard_ui.py`, and
browser code/styles in `web/`. Keep the source-only publication registry current
when adding assets. Rendering assembles and embeds the reviewed files with CSP
hashes; source syntax checks validate each script and the combined script.

```sh
.venv/bin/python tools/benchmark_dashboard.py --segments 4000 --iterations 3
.venv/bin/pytest -q tests/test_dashboard_metrics.py tests/test_local_dashboard.py tests/test_dashboard_explore.py
```

The benchmark generates synthetic intervals with exact collector-shaped support
runs and measures only calculation time. Disjoint support and real producer
integration are checked separately by the focused regression tests.
It does not open the ledger or load a model. Compare identical inputs on the same
machine and account for system load; it is not a model-throughput measurement.
For a paired comparison, `--baseline-ref REVIEWED_REVISION` selects calculation
functions from local Git history and checks output equality. Treat the revision
as executable code: choose only reviewed repository source. The tool does not
fetch source or import the old collector/controller; CI fetches its one pinned
public source baseline separately.

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
