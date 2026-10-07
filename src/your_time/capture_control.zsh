#!/bin/zsh
set -eu
umask 077

action=${1:-status}
agent_dir="$HOME/Library/LaunchAgents"
paused_dir="$HOME/Library/Application Support/personal-activity-ledger/paused-launchagents"
source_dir=/path/to/your-time/launchagents
domain="gui/$(id -u)"
labels=(com.yourtime.secure-screen-record com.yourtime.secure-mac-activity com.yourtime.secure-window-reader)

job_running() {
  launchctl print "$domain/$1" 2>/dev/null | /usr/bin/awk '
    /^[[:space:]]*state = running$/ { running = 1 }
    END { exit !running }'
}

case "$action" in
  pause)
    mkdir -p "$paused_dir"
    chmod 700 "$paused_dir"
    for label in "${labels[@]}"; do
      launchctl bootout "$domain/$label" 2>/dev/null || true
      if [[ -f "$agent_dir/$label.plist" ]]; then
        mv "$agent_dir/$label.plist" "$paused_dir/$label.plist"
      fi
    done
    print 'capture_paused'
    ;;
  resume)
    /opt/homebrew/bin/python3 -c 'import shutil,sys; sys.exit(0 if shutil.disk_usage("/path/to/home/Library/Application Support/personal-activity-ledger").free >= 10*1024**3 else 1)'
    for label in "${labels[@]}"; do
      if [[ -f "$paused_dir/$label.plist" ]]; then
        mv "$paused_dir/$label.plist" "$agent_dir/$label.plist"
      elif [[ ! -f "$agent_dir/$label.plist" ]]; then
        install -m 600 "$source_dir/$label.plist" "$agent_dir/$label.plist"
      fi
      chmod 600 "$agent_dir/$label.plist"
      launchctl print "$domain/$label" >/dev/null 2>&1 || launchctl bootstrap "$domain" "$agent_dir/$label.plist"
      # A low-disk worker can exit successfully while its agent stays loaded.
      # Bootstrap alone does not restart that service after space recovers.
      job_running "$label" || launchctl kickstart "$domain/$label"
    done
    print 'capture_resumed'
    ;;
  status)
    for label in "${labels[@]}"; do
      if job_running "$label"; then
        print "$label running"
      elif launchctl print "$domain/$label" >/dev/null 2>&1; then
        print "$label loaded_stopped"
      elif [[ -f "$paused_dir/$label.plist" ]]; then
        print "$label paused"
      else
        print "$label inactive"
      fi
    done
    ;;
  *)
    print -u2 'usage: capture_control.zsh pause|resume|status'
    exit 2
    ;;
esac
