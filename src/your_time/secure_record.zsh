#!/bin/zsh
set -eu
umask 077

# Refuse to launch capture when the disk guard would immediately stop it.
/opt/homebrew/bin/python3 -c 'import shutil,sys; sys.exit(0 if shutil.disk_usage("/path/to/home/Library/Application Support/personal-activity-ledger").free >= 10*1024**3 else 1)'

# A minimal environment and macOS sandbox deny all network connections.
exec /usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/secure_capture.py
