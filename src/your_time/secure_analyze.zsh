#!/bin/zsh
set -eu
umask 077

# Refresh recent complete days after Screen Time sync and overnight vision.
# This writes only to the private local activity root; network is denied.
/usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/daily_analysis.py \
  --days-ago 1 --days-ago 2 --days-ago 3 --write

/usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/daily_focus.py \
  --days-ago 1 --days-ago 2 --days-ago 3 --write

/bin/zsh /path/to/your-time/secure_dashboard.zsh
