#!/bin/zsh
set -eu
umask 077

# Bounded local 9B recovery for 2B-incomplete frames; no HTTP listener.
exec /usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/caffeinate -i \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/vision_fallback.py \
  "$@"
