#!/bin/zsh
set -eu
umask 077

# FSEvents metadata only within the two user-approved Projects roots.
exec /usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/project_file_watch.py
