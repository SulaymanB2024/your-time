#!/bin/zsh
set -eu
umask 077

# Refresh recent complete days after Screen Time sync and overnight vision.
# This writes only to the private local activity root; network is denied.
/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/daily_analysis.py \
  --days-ago 1 --days-ago 2 --days-ago 3 --write

/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/daily_focus.py \
  --days-ago 1 --days-ago 2 --days-ago 3 --write

/bin/zsh /Users/sulaymanbowles/Projects/personal-activity-ledger/secure_dashboard.zsh
