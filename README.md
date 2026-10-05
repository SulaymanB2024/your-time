# Your Time

A private, free activity chronicle for understanding daily behavior, focus and
progress. Mac foreground activity, iPhone app intervals and project metadata
feed an offline dashboard with day, week, month and year views. On-device 9B
models add contextual tags and summaries. Observations, model suggestions and
confirmed outcomes remain separate.

File-change batches stay queued until their SQLite transaction commits, and
transient write errors are retried. Empty flushes do not open the database.
The queue is bounded; abrupt process termination can still lose uncommitted
metadata. Setup checks distinguish a running watcher from paused writes.

Git scans filter the configured user's authored or committed changes before
applying the 500-commit per-store bound. Unreadable ref tips are counted while
valid history remains readable. Truncated histories and unreadable tips make
the scan explicitly partial; they are not a claim of complete project history.

This is the installed configuration for one Apple Silicon Mac. It runs locally,
without a web server, cloud inference, automatic uploads or paid services.
The public source repository contains no recorded activity or model weights.

## What runs

| Component | Collection / analysis | Schedule |
| --- | --- | --- |
| Mac sampler + signed window reader | Foreground app/window identity; locked, idle and unidentified intervals | Every 5 seconds |
| Screenshot collector | Changed visible screens; sensitive frontmost windows and locked sessions skipped | Checks every 10 seconds |
| Apple Vision OCR | Pending images and text geometry, without cloud services | Bounded daytime batches |
| Visual features | Local similarity fingerprints; tightly bounded context reuse | Every 30 minutes, resource gated |
| Signed iPhone reader | Apple's locally synced app-focus intervals; idempotent import | Hourly |
| Project file watcher | Path/time change metadata under approved project roots | Continuous FSEvents |
| Git reader | Local commit IDs/times attributable to configured identities; no messages/diffs | Daily |
| 9B vision | Diverse OCR-complete frames, recent incomplete cases and older weaker captions | 00:30–07:00 Chicago |
| 9B text | Window topics, chapter synthesis and visible-screen context | 07:00–08:00 Chicago |
| Dashboard | Private static HTML; updated context, timeline and historical totals | Every 5 minutes; opens at login |

Screenshot checks save an immediate frame after a window/app change; otherwise
at most one changed frame per 20 seconds, plus a context frame every two minutes.
Raw captures and model-selected frames are separate quantities. The model cannot
process every captured frame every night.

The supported scope is digital activity, not health tracking. Collection gaps
are unknown, not idle. Device totals can overlap and must not be added together.
A foreground window is not proof of attention. Files/commits can be produced by
automation and do not prove task completion. iPhone sync may lag and is not a
reconciliation with the phone's Screen Time UI.

## Private dashboard

Open `~/Library/Application Support/personal-activity-ledger/dashboard/index.html`
in the default browser. The login LaunchAgent opens it automatically after the
file exists. Refresh runs every five minutes; a stale tab refreshes on return.

The monochrome dashboard shows identified Mac time, iPhone app time, time by
supported task/context, a precise daily sequence, continuity distributions and
historical patterns. Context inspection exposes the observed labels and counted
intervals; unknown time remains visible. Recent days have detailed sequences;
all retained dates remain selectable for historical views. Source coverage is
available in a disclosure. Images and OCR bodies are not embedded.

The page contains private activity context. It is mode 0600 in a mode-0700
directory, uses no external assets and blocks network requests through its CSP.
Do not publish or upload the generated HTML.

## Runtime and permissions

Source: `/Users/sulaymanbowles/Projects/personal-activity-ledger`.
Data: `~/Library/Application Support/personal-activity-ledger`.
Dependencies are locked in `uv.lock` for Python 3.11 and macOS; recreate the
Python environment with `uv sync --locked` when intentionally restoring it.

LaunchAgent templates and wrappers contain this Mac's absolute paths. A clone
is source, not a portable one-command installer. A different Mac requires path
configuration, native builds and explicit macOS grants. Do not bulk load the
optional agents or grant permissions to the shared Python interpreter.

- `/Applications/YourTimeWindowReader.app` is locally signed and has Accessibility
  access. The Python sampler consumes only a fresh record from the same process.
- Capture has the existing Screen Recording grant.
- `/Applications/PhoneActivityReader.app` is the dedicated signed Full Disk Access
  reader. Screen Time sharing supplies Apple's local sync records.
- Phone packaging used PyInstaller 6.22.3 and `ActivityWatch/aw-import-screentime`
  commit `1297039793819b25f926289fb033d77c66786c50`; its parser dependency was
  pinned to `f1b796e9155974799fd9c2c84e0007b2016eb089`. These separate parser/build
  prerequisites are not included in this repository's Python runtime lock.
- Rebuilding a signed reader may invalidate its macOS permission. Verify the
  signature and refresh only that exact app's grant when needed.
- Calendar/Reminders and the custom browser extension are staged, optional
  integrations. They are not enabled. EventKit needs explicit grants and a
  selected-list allowlist before importing anything.

Collectors, OCR and model jobs run with a minimal environment under
`network-off.sb`. FileVault is enabled; runtime directories and files are
user-only. Stock ActivityWatch and Pensieve servers remain disabled. See
[SECURITY.md](SECURITY.md) for controls and limits.

## Overnight model analysis

The active model is pinned **Qwen3.5-9B Q8**, with an F16 vision projector, in
`vision_quality_model_manifest.json`. Local `llama-mtmd-cli` and
`llama-completion` 0.5.0 use Metal. There is no model HTTP server. Weights are
read-only under the private data root, outside Git.

`com.sulayman.overnight-vision` starts at 00:30 and retries every half hour
through 06:30. It calls the 9B worker through the legacy-named
`secure_vision_fallback.zsh`; **9B is the primary pass** and does not require a
2B caption. The 2B code is retained for manual comparison only. The removed
27B test weights must not be redownloaded as part of routine setup.
The scheduled launcher has no initial legacy-recovery reservation; eligible
frames with an earlier incomplete 2B result enter the ordinary primary queue.

Vision samples at most one OCR-complete frame per ten-second bucket, prioritizes
the latest complete day and spreads first attempts across that day. Its
2,000-frame ceiling is a safety bound, not a promised nightly quota. The 07:00
cutoff and measured throughput determine actual coverage.

Current-day candidates get at least nine slots before each older candidate
while current work remains. Older retries share that historical lane; they do
not run as an initial priority queue. Spare capacity can process history once
current-day work is exhausted. This is a selection policy, not a guaranteed
ratio of model time or successful descriptions.

The final hour, 07:00–08:00, is reserved for text tagging and synthesis, with a
07:30 retry. Shared locks prevent overlap between pipeline model processes.
Text stage budgets are 20, 20 and 15 minutes, clipped to 08:00. There are no
scheduled daytime LLM launches. Collection/OCR/dashboard updates continue.

Chapter analysis covers different local hours and prioritizes longer observed
blocks within each hour. It reserves one bounded call for a coherent summary
of at most 20 completed chapters. A valid summary can describe a partial day;
its support IDs, hashes and sampled seconds remain explicit. Its themes are
withheld if their supporting evidence changes. Matching chapter caches survive
summary-prompt updates, and a failed summary does not discard completed chapters.

Inference requires AC power, at least 10.5 GiB free, at least 20% free memory,
and acceptable system load. Workers recheck gates and deadlines between items,
including after sleep. They prevent idle sleep only while actively working;
the Mac must be awake at a launch trigger. Being loaded in launchd means
scheduled, not successfully completed.

Only final captions, tags, provenance and aggregate receipts are retained;
model reasoning is discarded. Labels require source support. Inference cannot
create a confirmed accomplishment. The earlier small direct image comparison
favored 9B; private evaluation artifacts stay local and do not establish an
accuracy guarantee across all tasks.

The current 9B latest receipt is `vision-fallback-latest-receipt.json`.
`vision-nightly-receipt.json` is a legacy 2B receipt, not proof of the current
9B run. Historical receipts and `vision_capacity.py` support throughput checks.

Every finished vision attempt, including a power/resource skip, now has an
immutable historical receipt. `text-nightly-latest-receipt.json` and
`text-run-receipts` record each text stage's return code, aggregate results and
elapsed time. A failed stage does not suppress the later stages or dashboard
refresh. Exit zero with a resource gate is not described as completed analysis.
The three scheduled text-stage argument lists are validated by regression tests.

The capture LaunchAgents also have five-minute start triggers, allowing an
existing loaded collector to retry after a successful low-storage stop.
Storage guards still apply on every retry. An explicit capture pause unloads
these jobs and disables their triggers until resumed.

## Setup checks and controls

From the source directory:

```sh
./.venv/bin/python setup_check.py --write
./.venv/bin/python setup_check.py --hash-models --write
./.venv/bin/python chronicle_audit.py --days 7 --write
```

The setup check is read-only apart from its optional private receipt. It checks
source/installed LaunchAgents, live collector freshness, signed readers,
permissions, SQLite integrity, dashboard hash, local binaries, network denial
and resource gates. `--hash-models` streams both model files for full SHA-256
verification; it does not run inference or download anything. It prints only
configuration/status metadata and aggregate counts, not activity bodies.
The data audit checks accounting bounds and joins, not semantic caption accuracy.

```sh
./capture_control.zsh status
./capture_control.zsh pause
./capture_control.zsh resume
```

Pause persists across logins. The storage guard pauses capture below 10 GiB and
never deletes source data. Resume requires enough free space. A skipped locked
or unchanged frame is normal. A successful phone import may still have stale
source events; check `phone-quality-latest.json` as well as the import receipt.
`status` distinguishes running services from loaded-but-stopped jobs. `resume`
also starts a loaded worker that exited when free space was low; it does not
interrupt already-running collectors.

`daily_export.py` previews by default; `--write` creates a private JSONL export
for a deliberate handoff. It does not upload. Correction labels and confirmed
outcomes are stored separately through `task_corrections.py`.

## Validation and source delivery

```sh
./.venv/bin/pytest -q
```

Tests use synthetic fixtures; live setup checks verify this installation.
For source-only updates to the public GitHub repository, use
[publish_source.py](publish_source.py) as described in [GITHUB.md](GITHUB.md).
Do not push the local historical working branch directly.
