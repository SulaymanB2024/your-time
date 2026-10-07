#!/bin/zsh
set -eu
umask 077

# Short-lived local text inference; no HTTP listener or external network.
/usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/caffeinate -i \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/overnight_text.py
