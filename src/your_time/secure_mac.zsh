#!/bin/zsh
set -eu
umask 077

exec /usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/mac_activity.py
