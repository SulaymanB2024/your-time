#!/bin/zsh
set -eu
umask 077

# Apple Vision OCR is local; network access is denied for the entire process.
exec /usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/index_screenshots.py \
  --limit 120 \
  --geometry-backfill-limit 80
