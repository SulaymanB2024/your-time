#!/bin/zsh
set -eu
umask 077

# A dedicated signed app receives only the permission needed to read Apple's
# Screen Time sync files. The macOS sandbox denies its network access.
/usr/bin/codesign --verify --deep --strict /Applications/PhoneActivityReader.app >/dev/null 2>&1
reader_status=0
/usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /Applications/PhoneActivityReader.app/Contents/MacOS/PhoneActivityReader || reader_status=$?

# Compare the readable Apple sync snapshot with the newest event already
# retained locally. This step needs no Full Disk Access and sends no data out.
/usr/bin/env -i \
  HOME=/path/to/home \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /path/to/your-time/network-off.sb \
  /path/to/your-time/.venv/bin/python \
  /path/to/your-time/phone_quality.py \
  --reader-exit "$reader_status"

/bin/zsh /path/to/your-time/secure_dashboard.zsh
exit "$reader_status"
