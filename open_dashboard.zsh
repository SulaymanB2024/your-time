#!/bin/zsh
set -eu

dashboard="$HOME/Library/Application Support/personal-activity-ledger/dashboard/index.html"

# The refresh agent also starts at login. Wait briefly for its first render.
for ((attempt = 0; attempt < 60; attempt++)); do
  if [[ -s "$dashboard" ]]; then
    /usr/bin/open "$dashboard"
    exit 0
  fi
  /bin/sleep 1
done

exit 1
