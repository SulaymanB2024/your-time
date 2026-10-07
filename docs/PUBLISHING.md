# Source repository

The public GitHub repository is `SulaymanB2024/your-time`. It contains code,
tests, configuration templates, dependency locks and these technical guides.
It does not contain the recorded database, screenshots, OCR, captions,
generated dashboard, exports, model weights or private review notes.

## Two separate histories

The running checkout retains its original local history on
`codex/secure-local-capture`. Some older commits include private review notes.
**Do not push that branch, other local branches, tags, or `--mirror`.**

The source publisher builds a clean source tree from committed files in a
separate temporary index. Its initial commit has no parents. Later source
commits descend only from the previously published source commit. The live
checkout, local history, working index and running collectors stay in place.
The published line is recorded locally as `codex/github-source`; GitHub uses
`main`. Default `git push` is disabled locally with `push.default=nothing`.
The installed local pre-push hook also rejects extra refs, deletions, excluded
files and commits with private ancestry. The hook invokes this checkout's
`publish_source.py --check-push`; it is a local guard and is not automatically
installed by cloning GitHub.

## Publish an intentional source update

Commit reviewed changes locally first. Preview the source export:

```sh
make source-preview
```

After reviewing the change and running relevant tests, explicitly push:

```sh
make publish
```

The exporter verifies the exact repository, public visibility and previous
published commit. It pushes only the source commit to `main`, without force.
It uses a file allowlist, rejects links/binary payloads and checks common
credential and embedded-image patterns. These checks reduce accidental
disclosure; they are not a guarantee that arbitrary text is safe to share.
Review source diffs before publishing. Personal review documents are excluded
even if tracked in the local checkout.

Source modules are registered by filename. New code or documentation must
be deliberately added to the publication policy; arbitrary Python/TOML files
are excluded. Regression tests are published from `tests/`, and only the named
public guides from `docs/`. `tools/check.py` and `Makefile` provide the source
quality gate. Run `make check` before publication.

## Public layout and privacy gate

Workers and adjacent runtime resources live under `src/your_time/`. Mac templates
are under `macos/`, optional browser integration under `integrations/`, and
publication utilities under `tools/`. Tests, frontend assets and engineering guides
have separate directories. The installed checkout retains its stable launcher
layout; publication maps reviewed files into these folders without moving jobs.

The publisher reads an optional private installation-redaction policy outside
Git. It applies explicit substitutions to filenames and bodies, then refuses
remaining concrete home paths, non-example contact information, credential
patterns, images, binary payloads and unregistered runtime files. Synthetic example
contacts and intentionally public GitHub attribution are permitted. Duplicate
destinations and linked source entries refuse publication. The organized export
is tested independently; frozen installed source hashes remain unchanged.

A clean latest tree does not erase earlier commits, pull-request refs, branches,
hosted caches or external clones. Historical removal requires separately approved
cleanup. Publication never force-pushes or silently rewrites history.

The repository creates no hosted dashboard, cloud inference, recurring
publisher or paid CI jobs. Runtime data remains exclusively on the Mac.
The user approved source-only hosted CI; see [CI.md](CI.md). Publication
requires GitHub workflow permission, and a passing hosted run must be checked
separately. The user also approved enabling this public repository's
secret-scanning alerts, secret push protection, vulnerability alerts and
Dependabot security updates. Their enabled settings were read back after the
change. Built-in pattern detection complements deliberate source review; it
does not guarantee that all private material is detected.
