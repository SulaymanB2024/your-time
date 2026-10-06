"""Run bounded text stages and preserve content-free evidence of each attempt."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from overnight_schedule import text_budget
from private_io import open_private_file, prepare_directory
from secure_store import STATE_DIR, prepare_private_dir
from setup_check import write_private

PROJECT = Path(__file__).resolve().parent
STAGES = (
    ('window_topics', 'window_topic_tagging.py', ['--overnight-only', '--max-seconds', '1200',
      '--days-ago', '1', '--days-ago', '2', '--days-ago', '3', '--days-ago', '0']),
    ('chapters', 'local_synthesis.py', ['--overnight-only', '--max-seconds', '1200']),
    ('screen_context', 'screen_context_tagging.py', ['--overnight-only', '--max-seconds', '900',
      '--days-ago', '1', '--days-ago', '2', '--days-ago', '0']),
    ('similarity', 'screen_similarity.py', []),
    ('dashboard', 'local_dashboard.py', []),
)
COUNTS = {'selected', 'tagged', 'retained_total', 'processed_this_run', 'specific',
          'sensitive_skipped', 'eligible_titles', 'tagged_titles', 'specific_titles',
          'total_blocks', 'complete_blocks', 'attempted_this_run', 'completed_this_run',
          'eligible_blocks', 'summary_evidence_blocks',
          'summary_input_blocks', 'summary_cited_blocks',
          'insufficient_context_blocks', 'reused', 'failed_this_run', 'propagated', 'days',
          'candidate_frames', 'clusters', 'supported_clusters', 'conflicting_clusters', 'propagated_count'}
TIMINGS = {'summary_input_seconds', 'summary_cited_seconds'}
GOOD = {'complete', 'up_to_date', 'private_dashboard_written'}
CODE = re.compile(r'[a-zA-Z0-9_+-]{1,80}\Z')


def output_summaries(stdout: str) -> list[dict]:
    summaries = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        row = {}
        for key in ('status', 'stop_reason', 'reason', 'summary_status',
                    'summary_stop_reason', 'day_failure_code'):
            item = value.get(key)
            if item is None or isinstance(item, str) and CODE.fullmatch(item):
                if key in value:
                    row[key] = item
        for key in COUNTS:
            item = value.get(key)
            if type(item) is int and item >= 0:
                row[key] = item
        for key in TIMINGS:
            item = value.get(key)
            if type(item) in (int, float) and math.isfinite(item) and item >= 0:
                row[key] = item
        if row:
            summaries.append(row)
    return summaries


def save(report: dict) -> None:
    prepare_private_dir()
    write_private(report, STATE_DIR / 'text-nightly-latest-receipt.json')
    if report.get('finished_at_utc'):
        history = STATE_DIR / 'text-run-receipts'
        prepare_directory(history)
        stamp = report['started_at_utc'].replace(':', '-').replace('+', '_')
        path = history / ('text-' + stamp + '.json')
        if path.exists():
            raise RuntimeError('Text historical receipt already exists')
        write_private(report, path)


def run() -> dict:
    started = time.monotonic()
    report = {'schema_version': 1, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'running', 'stages': [], 'stop_reason': None}
    if text_budget(210) < 210:
        report.update(status='skipped', stop_reason='outside_text_window')
    else:
        save(report)
        for name, script, args in STAGES:
            if name in {'window_topics', 'chapters', 'screen_context'} and text_budget(210) < 210:
                report['stages'].append({'name': name, 'status': 'skipped',
                                         'stop_reason': 'text_window_ended'})
                continue
            stage_started = time.monotonic()
            row = {'name': name}
            try:
                # Workers enforce their own inference deadlines. The inherited
                # sandbox/minimal environment blocks all children from networking.
                result = subprocess.run([sys.executable, str(PROJECT / script), *args],
                                        capture_output=True, text=True, check=False)
                summaries = output_summaries(result.stdout)
                successful = summaries and (all(r.get('status') in GOOD for r in summaries) or
                    name == 'similarity' and all('propagated_count' in r for r in summaries))
                row.update(exit_code=result.returncode, results=summaries,
                           status='failed' if result.returncode else
                           'complete' if successful
                           else 'partial')
            except OSError:
                row.update(status='failed', stop_reason='worker_launch_failed')
            row['elapsed_seconds'] = round(time.monotonic() - stage_started, 2)
            report['stages'].append(row)
            save(report)
        report['status'] = 'complete' if all(r['status'] == 'complete' for r in report['stages']) else 'partial'
    report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(time.monotonic() - started, 2))
    save(report)
    return report


def main() -> None:
    os.umask(0o077)
    prepare_private_dir()
    fd = open_private_file(STATE_DIR / 'text-nightly-controller.lock')
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status': 'another_text_controller_is_running'}))
            return
        report = run()
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(1 if any(r['status'] == 'failed' for r in report['stages']) else 0)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
