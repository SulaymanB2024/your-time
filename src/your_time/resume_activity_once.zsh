#!/bin/zsh
set -eu
umask 077

# Installed as a one-time, date-gated LaunchAgent when the user asks to pause
# this ledger temporarily. It resumes only this repository's known jobs.
phase=${1:?capture or daytime phase required}
not_before=${2:?local YYYY-MM-DD date required}
[[ "$not_before" == <->-<->-<-> ]] || exit 2
[[ "$(/bin/date +%Y-%m-%d)" < "$not_before" ]] && exit 0

agent_dir="$HOME/Library/LaunchAgents"
paused_dir="$HOME/Library/Application Support/personal-activity-ledger/paused-launchagents"
receipt_dir="$HOME/Library/Application Support/personal-activity-ledger"
domain="gui/$(id -u)"
label="com.yourtime.activity-resume-${phase}-once"

case "$phase" in
  capture)
    /bin/zsh /path/to/your-time/capture_control.zsh resume
    ;;
  daytime)
    labels=(
      com.yourtime.activity-dashboard-open
      com.yourtime.activity-dashboard-refresh
      com.yourtime.daily-analysis
      com.yourtime.project-file-watch
      com.yourtime.project-git
      com.yourtime.screen-storage-guard
      com.yourtime.secure-features
      com.yourtime.secure-offline-ocr
      com.yourtime.secure-phone-import
      com.yourtime.secure-text-analysis
    )
    for job in "${labels[@]}"; do
      if [[ -f "$paused_dir/$job.plist" ]]; then
        mv "$paused_dir/$job.plist" "$agent_dir/$job.plist"
      fi
      if [[ -f "$agent_dir/$job.plist" ]]; then
        chmod 600 "$agent_dir/$job.plist"
        launchctl print "$domain/$job" >/dev/null 2>&1 || \
          launchctl bootstrap "$domain" "$agent_dir/$job.plist"
      fi
    done
    ;;
  *) exit 2 ;;
esac

# Mark completion before unloading this one-shot job. A later invocation is
# harmless if launchd has not yet removed the loaded service.
print -r -- "$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ) $phase resumed" > "$receipt_dir/resume-${phase}-receipt.txt"
chmod 600 "$receipt_dir/resume-${phase}-receipt.txt"
rm -f "$agent_dir/$label.plist"
launchctl bootout "$domain/$label" >/dev/null 2>&1 || true
