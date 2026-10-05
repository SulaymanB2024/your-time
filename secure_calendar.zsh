#!/bin/zsh
set -eu
umask 077

root="$HOME/Library/Application Support/personal-activity-ledger"
scope="$root/calendar-scope.json"
exported="$root/calendar-eventkit-export.json"

# The signed local reader is network-blocked and exports only selected lists.
/usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Applications/YourTimeCalendarReader.app/Contents/MacOS/calendar-reader \
  --scope "$scope" --output "$exported" >/dev/null

/usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 \
  /usr/bin/sandbox-exec -f /Users/sulaymanbowles/Projects/personal-activity-ledger/network-off.sb \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/.venv/bin/python \
  /Users/sulaymanbowles/Projects/personal-activity-ledger/calendar_import.py
