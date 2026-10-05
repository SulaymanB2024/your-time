#!/bin/zsh
set -eu
umask 077

# Rebuild one private offline file; no web server and no network.
exec /usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/local_dashboard.py
