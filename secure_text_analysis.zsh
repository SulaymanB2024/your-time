#!/bin/zsh
set -eu
umask 077

# Short-lived local text inference; no HTTP listener or external network.
/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/caffeinate -i \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/window_topic_tagging.py \
  --overnight-only --max-seconds 1200 --days-ago 1 --days-ago 2 --days-ago 3 --days-ago 0

/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/caffeinate -i \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/local_synthesis.py \
  --overnight-only --max-seconds 1200

/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/caffeinate -i \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/screen_context_tagging.py \
  --overnight-only --max-seconds 900 --days-ago 1 --days-ago 2 --days-ago 0

/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/screen_similarity.py

/bin/zsh /Users/sulaymanbowles/Projects/personal-activity-ledger/secure_dashboard.zsh
