#!/bin/zsh
set -eu
umask 077

# Pinned 9B vision owns 00:30–07:00.
# It runs network-blocked with no HTTP listener, checks power and resources
# between frames, and stops by 07:00, reserving 07:00–08:00 for text analysis.
/usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/caffeinate -i \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/specialization_study.py run || true

/bin/zsh /path/to/your-time/secure_vision_fallback.zsh \
  --limit 2000 --mode mixed --hard-limit 0
