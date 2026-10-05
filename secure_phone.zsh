#!/bin/zsh
set -eu
umask 077

# A dedicated signed app receives only the permission needed to read Apple's
# Screen Time sync files. The macOS sandbox denies its network access.
/usr/bin/codesign --verify --deep --strict /Applications/PhoneActivityReader.app >/dev/null 2>&1
reader_status=0
/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Applications/PhoneActivityReader.app/Contents/MacOS/PhoneActivityReader || reader_status=$?

# Compare the readable Apple sync snapshot with the newest event already
# retained locally. This step needs no Full Disk Access and sends no data out.
/usr/bin/env -i \
  HOME=/Users/sulaymanbowles \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec \
  -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/phone_quality.py \
  --reader-exit "$reader_status"

/bin/zsh /Users/sulaymanbowles/Projects/personal-activity-ledger/secure_dashboard.zsh
exit "$reader_status"
