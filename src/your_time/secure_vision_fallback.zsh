#!/bin/zsh
set -eu
umask 077

# Primary local 9B analysis, bounded by the overnight window; no HTTP listener.
exec /usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/caffeinate -i \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/vision_fallback.py \
  "$@"
