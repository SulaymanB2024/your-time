"""Install a process timer that survives exec and loss of the coordinator."""
from __future__ import annotations

import math
import os
import signal
import sys


def main() -> None:
    seconds = float(sys.argv[1])
    command = sys.argv[2:]
    if not command or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive deadline and command required')
    # exec preserves ITIMER_REAL. The native child retains this timer and the
    # inherited model lock even if the Python coordinator is abruptly killed.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
