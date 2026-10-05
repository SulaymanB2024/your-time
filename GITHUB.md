# Source repository

The private GitHub repository is `SulaymanB2024/your-time`. It contains code,
tests, configuration templates, dependency locks and these technical guides.
It does not contain the recorded database, screenshots, OCR, captions,
generated dashboard, exports, model weights or private review notes.

## Two separate histories

The running checkout retains its original local history on
`codex/secure-local-capture`. Some older commits include private review notes.
**Do not push that branch, other local branches, tags, or `--mirror`.**

`publish_source.py` builds a clean source tree from committed files in a
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
./.venv/bin/python publish_source.py
```

After reviewing the change and running relevant tests, explicitly push:

```sh
./.venv/bin/python publish_source.py --push
```

The exporter verifies the exact repository, private visibility and previous
published commit. It pushes only the source commit to `main`, without force.
It uses a file allowlist, rejects links/binary payloads and checks common
credential and embedded-image patterns. These checks reduce accidental
disclosure; they are not a guarantee that arbitrary text is safe to share.
Review source diffs before publishing. Personal review documents are excluded
even if tracked in the local checkout.

The repository creates no hosted dashboard, cloud inference, recurring
publisher or paid CI jobs. Runtime data remains exclusively on the Mac.
