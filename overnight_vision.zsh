#!/bin/zsh
set -eu
umask 077

# Pinned 9B vision owns 00:30–07:00.
# It runs network-blocked with no HTTP listener, checks power and resources
# between frames, and stops by 07:00, reserving 07:00–08:00 for text analysis.
/bin/zsh /Users/sulaymanbowles/Projects/personal-activity-ledger/secure_vision_fallback.zsh \
  --limit 2000 --mode mixed --hard-limit 0
