#!/bin/zsh
set -eu
umask 077

# Apple's Vision feature prints stay on-device. This worker never connects out.
exec /usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/index_features.py \
  --limit 120 --allow-battery
