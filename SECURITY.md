# Security boundary

The archive contains screen images, OCR text, app names, URLs in older
Pensieve images, and iPhone use times. Treat the entire data root as sensitive.

## Controls in effect

- FileVault protects the disk while the Mac is powered off. The login lock
  limits interactive access, but running same-user processes remain a risk.
- The activity root, screenshot directories, source database, and config are
  user-only. New jobs set `umask 077`; database and image files are mode 0600.
- Capture, Mac sampling, OCR, and the packaged phone reader use a macOS
  `sandbox-exec` profile denying all network operations. They have a minimal
  environment with no credentials or telemetry tokens.
- The active collectors expose **no HTTP listener**. Stock ActivityWatch and
  Pensieve servers, plus the ActivityWatch Chrome extension path, are disabled.
- The local JSONL export requires `--write`; nothing automatically sends data
  to an external model, Drive, or another service. The staged Chrome extension,
  once reviewed and enabled, sends bounded active-tab metadata only to a
  network-blocked native host over stdio. Its review popup receives interval
  times and task labels, never screenshots or OCR.
- The overnight vision job uses the pinned local 9B Q8 model as its primary
  pass through short-lived CLI processes, with no model server. The 2B model
  is available only for manual comparison. Model processes have network access
  denied and share a lock with the text pipeline. They store only final captions, not generated
  reasoning, and treat screenshot text as untrusted input with no tools to invoke.
- A tested storage guard pauses capture below 10 GiB free and preserves the
  recorded source files.
- The recorder skips a locked or unknown session, configured sensitive
  frontmost apps/windows, and redundant frames. Mac app timing continues
  separately while a screenshot is skipped.
- Daily analysis is a network-blocked local job. Its private reports label
  sample gaps and keep vision descriptions separate from observed usage.
- Text-only 9B synthesis uses the same pinned local model weights through a
  network-denied CLI with constrained JSON output. It receives bounded
  summaries of private focus blocks, has no tools, and cannot mark a model
  suggestion as a confirmed accomplishment. Its private prompt is passed via
  a temporary mode-0600 file in a mode-0700 directory, not in process arguments;
  the temporary file is removed after the model call.
- Window-topic tagging processes private window titles and selected stronger
  captions in bounded batches. It retains only topic labels and hashed window
  identities, rejects specific labels without source-word support, and keeps
  broader app/site context explicitly marked as such. It has no network or
  tool access.
- OCR text geometry and Apple Vision feature prints are kept in the private
  SQLite database. Similarity propagation requires a close match on the same
  app/display within two minutes and never promotes a model label to a
  verified accomplishment. The raw screenshot archive is unchanged.
- Git scanning is read-only, network-blocked, and restricted to user-approved
  project roots. It stores commit IDs/times attributable to locally configured
  user identities, not diffs or commit messages. The FSEvents watcher is also
  network-blocked and stores path/time metadata only after excluding generated
  dependency/build/hidden trees; paths remain sensitive and user-only.
- Browser tab capture, if enabled, discards URL paths, queries, fragments,
  credentials, and incognito activity in the extension before native messaging.
  The native host validates message size, timestamp, origin allowlist through
  Chrome's host manifest, domain, and correction IDs. It exposes no HTTP port.
- User correction labels are stored separately from model suggestions and are
  retractable. The staged EventKit reader is locally signed but has not been
  granted Calendar or Reminders access. EventKit grants full source access by
  design; a private allowlist must be selected before any event import, and
  calendar/reminder titles are off by default.
- The offline dashboard is a user-only local HTML file, rebuilt atomically.
  It embeds aggregate and activity-context data but no raw screenshots or OCR,
  loads no external assets, and blocks network requests with a content security
  policy. It exposes no listener.
- The phone reader is a dedicated ad-hoc-signed app with a verified bundle
  signature. The user granted it Full Disk Access after an exact-scope prompt;
  its hourly launchd job is network-blocked and has been verified live.
- A separate one-time signed backfill build read older plausible Apple sync
  intervals under the network-denying sandbox. It did not replace or expand
  the hourly reader's permissions. Its runtime build is a temporary artifact.

## Why the stock servers remain off

ActivityWatch 0.13.2 has no API authentication. Its own security guide says
this is unsafe on a multi-user Mac. Pensieve 0.37.0 has permissive CORS, an
unauthenticated file route that accepts arbitrary absolute paths, and a
configuration API able to change processing settings. Its server can also
write screenshot thumbnails into shared `/tmp` and enable Logfire telemetry
if a token is present. These are unacceptable for an always-on private archive.

The prior databases and screenshots were preserved. All current
source directories and files were tightened to user-only permissions. Stock
web services must not be restarted for routine review.

## Private file handling

`private_io.py` centralizes private directory traversal, file writes and lock
descriptors. Directory components and file opens reject symbolic links; file
targets must be owned regular files without additional hard links. Atomic
writes use a random exclusively created temporary file, flush it, and replace
the destination relative to an open directory descriptor. Failed writes retain
the previous complete output and remove the temporary file. Private permissions
are 0700 for directories and 0600 for newly created files.

The store creates the private database before SQLite opens it and validates
existing WAL, SHM and journal sidecars. Connections no longer change a
process-wide umask, and failed transactions roll back. These checks protect
against accidental path redirection and unsafe pre-existing filesystem entries;
they do not protect a user account that a hostile same-user process controls.

The Python source and the separately packaged, signed readers have different
release lifecycles. Changing a source helper does not upgrade the installed
reader bundle. Keep those bundles pinned until a separate build, signing and
privacy-permission readback is authorized. Signature validity alone does not
establish that an arbitrary newly signed bundle is the previously reviewed code.

## Source publication boundary

The public GitHub repository receives only an intentional source export.
Generated HTML, database files, images, OCR, captions, model weights, runtime
receipts and private review documents remain local. The initial source commit
has no ancestry from the local historical working branch; subsequent exports
have only published source parents. Default pushing is disabled on the working
checkout. See [GITHUB.md](GITHUB.md). Ignore rules and pattern checks are a
publication guard, not encryption or a substitute for source review.

Root modules are explicitly registered; arbitrary new Python or TOML files are
excluded. Tests and the three public engineering guides have bounded publication
locations. The exporter checks common GitHub/OpenAI, AWS, Google, GitLab,
Hugging Face, Slack and private-key patterns and rejects symlinks, non-file
objects and embedded images. It still cannot classify all private prose or
every credential format; a deliberate source review remains required.

## Verification

`make check` runs source syntax checks, Ruff and synthetic regression tests.
The filesystem tests attempt symlink/hard-link redirection, linked parents,
unsafe locks and database sidecars, interrupted writes, concurrent atomic
replacement, private SQLite sidecars and transaction rollback. `make setup`
checks the installed Mac separately, including network denial and signatures.
`make audit` queries package/version advisory metadata without activity data.

The October 2026 review upgraded the development-only pytest dependency from
8.4.2 to 9.0.3 for its temporary-directory handling advisory:
[pytest's patched release](https://github.com/pytest-dev/pytest/releases/tag/9.0.3).
An advisory scan is limited to the inspected environment and currently
published advisories; separately packaged reader dependencies are outside it.

## Remaining limits

- Any unsandboxed process already running as this macOS user can read
  user-accessible plaintext files while the session is unlocked. FileVault
  does not solve a live same-user compromise.
- Screenshots can contain passwords, financial information, other people's
  messages, and notifications. The frontmost-app skip rule does not inspect
  or exclude sensitive background windows or browser page contents. Capture
  covers all visible displays and does not claim reliable secret redaction.
  Use `capture_control.zsh pause` before particularly sensitive work.
- The local vision prompt and redaction are best-effort. A small model can
  hallucinate visible activity or repeat sensitive text. Descriptions are
  labeled as untrusted inference and are not automatically shared externally.
- If the custom Chrome extension is approved, its `tabs` permission can read
  active-tab titles and URLs before it discards URL details. Keep the unpacked
  source and Native Messaging allowlist pinned and review updates before use.
- If EventKit access is approved, macOS grants the reader broad full access
  even though this pipeline retains only selected lists. Excluding health
  calendars and sensitive titles is best-effort, so source selection matters.
- The dashboard itself contains sensitive app usage and window context. A
  browser rendering the local file may retain ordinary browser history or
  cache; use the private local copy and do not upload or publish it. Text-model
  themes remain inferences, even when they cite recorded blocks.
- The existing Ollama app and model are outside this setup. Its pre-existing
  local server is not used to process private screenshots.
- The `sandbox-exec` network rule was tested on this macOS version. It is a
  legacy macOS interface; recheck it after OS upgrades. It complements the
  absence of network calls in the running capture/OCR code.
- The dedicated phone app is locally signed, not notarized by Apple. Its
  bundle signature is checked before each scheduled import, and its network
  access is denied. Its Full Disk Access permission is broad by nature; keep
  the installed bundle pinned and repeat review after any rebuild.
- The Mac window reader is a separate locally signed app. Its Accessibility
  grant can read focused UI text across apps, so the installed code must stay
  pinned. It runs directly under launchd inside the network block, samples
  only the frontmost process, and writes a bounded focused title and browser
  hostname to one mode-0600 local file. The Mac sampler requires a fresh sample
  from the same process ID. The shared Python interpreter has no Accessibility
  grant; do not enable its disabled macOS entry.
- Apple Screen Time sync can lag or omit events. Intervals over six hours are
  excluded as suspect rather than counted as usage. The hourly import now runs
  under launchd, but ongoing completeness requires later source freshness
  readback.
- Locked-screen skipping and fresh unlocked capture were both verified on the
  live Mac. The new screenshots were OCR-indexed with nearby app context
  without exposing their contents in this conversation.

Primary vendor guidance: [ActivityWatch security](https://docs.activitywatch.net/en/latest/security.html),
[Apple App Sandbox](https://developer.apple.com/documentation/security/protecting-user-data-with-app-sandbox),
[Apple ScreenCaptureKit permissions](https://developer.apple.com/documentation/screencapturekit).
