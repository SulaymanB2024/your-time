"""Inspect this Mac's installed setup without recording or exposing activity.

Default: read-only, metadata-only JSON. --write saves a private receipt.
--hash-models additionally verifies the active 9B weights and projector; it
never downloads weights, starts inference, changes permissions or loads jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from overnight_schedule import START, VISION_END, END

PROJECT = Path(__file__).resolve().parent
STATE = Path.home() / 'Library/Application Support/personal-activity-ledger'
OPTIONAL_JOBS = {'com.sulayman.secure-calendar'}
PERSISTENT_JOBS = {'com.sulayman.secure-window-reader', 'com.sulayman.secure-mac-activity',
                   'com.sulayman.secure-screen-record', 'com.sulayman.project-file-watch'}
RECEIPTS = {
    'mac-activity-status.json': 120,
    'window-reader-latest.json': 30,
    'screen-capture-status.json': 120,
    'screen-storage-status.json': 180,
    'phone-latest-receipt.json': 90 * 60,
    'phone-quality-latest.json': 90 * 60,
    'ocr-latest-receipt.json': 30 * 60,
    'screen-feature-latest-receipt.json': 60 * 60,
    'project-file-latest-receipt.json': 120,
    'project-git-latest-receipt.json': 26 * 3600,
    'dashboard/manifest.json': 15 * 60,
}
SAFE_FIELDS = {'status', 'quality_status', 'stop_reason', 'capture_blocked_reason',
               'window_identity_status', 'accessibility_access', 'screen_capture_access',
               'trusted', 'below_threshold', 'selected', 'completed', 'failed',
               'sensitive_skipped', 'elapsed_seconds', 'errors', 'dropped',
               'pending_after_run', 'reader_exit_code'}
SAFE_CODE = re.compile(r'[a-zA-Z0-9_+-]{1,80}\Z')


def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15, check=False,
                              env={'HOME': str(Path.home()), 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin',
                                   'LANG': 'en_US.UTF-8'})
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(args, 124, '', '')


def private_path(path: Path) -> bool:
    try:
        stat = path.stat()
        return not path.is_symlink() and stat.st_uid == os.getuid() and not stat.st_mode & 0o077
    except OSError:
        return False


def receipt_summary(path: Path, now: datetime, max_age: float) -> dict:
    result = {'exists': path.is_file(), 'private': private_path(path), 'fresh': False}
    if not result['exists']:
        return result
    try:
        value = json.loads(path.read_text())
        stamp = next(value[k] for k in ('checked_at_utc', 'sampled_at_utc', 'generated_at_utc') if value.get(k))
        age = (now - datetime.fromisoformat(stamp)).total_seconds()
        result.update(age_seconds=round(age, 1), fresh=-30 <= age <= max_age)
        # Only known status codes, numbers and booleans enter this receipt.
        result['fields'] = {k: v for k, v in value.items() if k in SAFE_FIELDS and
                            (v is None or isinstance(v, (bool, int, float)) or
                             isinstance(v, str) and SAFE_CODE.fullmatch(v))}
    except (OSError, ValueError, TypeError, StopIteration, AttributeError):
        result['invalid'] = True
    return result


def inspect_job(source: Path, installed: Path) -> dict:
    expected = plistlib.loads(source.read_bytes())
    label = expected['Label']
    output = command(['/bin/launchctl', 'print', f'gui/{os.getuid()}/{label}'])
    result = {'loaded': output.returncode == 0, 'installed': installed.is_file(),
              'matches_source': False, 'private': private_path(installed)}
    if installed.is_file():
        try:
            result['matches_source'] = plistlib.loads(installed.read_bytes()) == expected
        except (ValueError, OSError):
            pass
    match = re.search(r'\bstate = ([^\n]+)', output.stdout)
    result['state'] = match.group(1).strip() if match else None
    match = re.search(r'last exit code = (\d+)', output.stdout)
    result['last_exit_code'] = int(match.group(1)) if match else None
    result['arguments_loaded'] = output.returncode == 0 and all(
        arg in output.stdout for arg in expected['ProgramArguments'])
    # A loaded calendar job is scheduled, not proof of completed inference.
    result['scheduled'] = 'StartCalendarInterval' in expected
    return result


def model_files(state: Path, project: Path, hash_models: bool) -> dict:
    try:
        manifest = json.loads((project / 'vision_quality_model_manifest.json').read_text())
        if {item['name'] for item in manifest['files']} != {'Qwen3.5-9B-Q8_0.gguf', 'mmproj-F16.gguf'}:
            raise ValueError('Unexpected active model files')
    except (OSError, ValueError, TypeError, KeyError):
        return {'active_model': 'Qwen3.5-9B-Q8_0', 'hashes_checked': False, 'files': [],
                'vision_cli_exists': False, 'text_cli_exists': False, 'manifest_invalid': True}
    folder = state / 'models/qwen3.5-9b-q8-eval'
    rows = []
    for item in manifest['files']:
        path = folder / item['name']
        row = {'name': item['name'], 'exists': path.is_file(), 'private': private_path(path),
               'size_matches': path.is_file() and path.stat().st_size == item['size']}
        if hash_models and row['exists'] and not path.is_symlink():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(chunk)
            row['sha256_matches'] = digest.hexdigest() == item['sha256']
        rows.append(row)
    return {'active_model': 'Qwen3.5-9B-Q8_0', 'hashes_checked': hash_models, 'files': rows,
            'vision_cli_exists': Path('/opt/homebrew/Cellar/llama.cpp/0.5.0/bin/llama-mtmd-cli').is_file(),
            'text_cli_exists': Path('/opt/homebrew/Cellar/llama.cpp/0.5.0/bin/llama-completion').is_file()}


def probe_network_block(project: Path) -> bool:
    code = """import errno,socket,sys
blocked=False
try:
    s=socket.create_connection(('127.0.0.1',9),timeout=0.5); s.close()
except OSError as e:
    blocked=e.errno in (errno.EPERM,errno.EACCES)
sys.exit(0 if blocked else 1)
"""
    return command(['/usr/bin/sandbox-exec', '-f', str(project / 'network-off.sb'),
                    str(project / '.venv/bin/python'), '-c', code]).returncode == 0


def database_health(state: Path) -> dict:
    path = state / 'activity-ledger.sqlite3'
    result = {'exists': path.is_file(), 'private': private_path(path), 'quick_check_ok': False}
    if not path.is_file():
        return result
    db = None
    try:
        db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
        with db:
            result['quick_check_ok'] = db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
            result['counts'] = {name: db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
                                for name in ('events', 'screenshots', 'vision_descriptions',
                                             'git_receipts', 'project_file_events')}
    except sqlite3.Error:
        result['read_failed'] = True
    finally:
        if db is not None:
            db.close()
    return result


def dashboard_health(state: Path) -> bool:
    try:
        manifest = json.loads((state / 'dashboard/manifest.json').read_text())
        payload = (state / 'dashboard/index.html').read_bytes()
        return (len(payload) == manifest['bytes'] and
                hashlib.sha256(payload).hexdigest() == manifest['sha256'] and
                private_path(state / 'dashboard/index.html'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def inspect(*, hash_models: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    failures, warnings = [], []
    jobs = {}
    for source in sorted((PROJECT / 'launchagents').glob('*.plist')):
        label = plistlib.loads(source.read_bytes())['Label']
        installed = Path.home() / 'Library/LaunchAgents' / source.name
        if label in OPTIONAL_JOBS and not installed.is_file():
            jobs[label] = {'optional': True, 'installed': False, 'loaded': False}
            continue
        result = inspect_job(source, installed)
        jobs[label] = result
        if not all(result[k] for k in ('loaded', 'installed', 'matches_source', 'private', 'arguments_loaded')):
            failures.append('job_setup:' + label)
        if label in PERSISTENT_JOBS and result['state'] != 'running':
            failures.append('collector_not_running:' + label)
        if result['last_exit_code'] not in (None, 0):
            warnings.append('job_last_exit_nonzero:' + label)
    receipts = {name: receipt_summary(STATE / name, now, age) for name, age in RECEIPTS.items()}
    for name, result in receipts.items():
        if not result['exists'] or not result['private'] or result.get('invalid'):
            failures.append('receipt_missing_or_unprotected:' + name)
        elif not result['fresh']:
            warnings.append('receipt_stale:' + name)
    mac = receipts['mac-activity-status.json'].get('fields', {})
    reader = receipts['window-reader-latest.json'].get('fields', {})
    if mac.get('accessibility_access') is not True or reader.get('trusted') is not True:
        failures.append('window_reader_access_denied')
    if mac.get('screen_capture_access') is not True:
        failures.append('screen_recording_access_denied')
    if mac.get('window_identity_status') != 'normal':
        warnings.append('window_identity_degraded')
    if receipts['phone-quality-latest.json'].get('fields', {}).get('reader_exit_code') != 0:
        warnings.append('phone_import_failed')
    if receipts['project-git-latest-receipt.json'].get('fields', {}).get('status') == 'partial':
        warnings.append('git_metadata_partial')
    database = database_health(STATE)
    if not database['private'] or not database['quick_check_ok']:
        failures.append('database_integrity_or_permissions')
    directories = {name: private_path(STATE / name) for name in ('', 'screenshots', 'analyses', 'dashboard', 'models')}
    if not all(directories.values()):
        failures.append('private_directory_permissions')
    dashboard_ok = dashboard_health(STATE)
    if not dashboard_ok or not private_path(STATE / 'dashboard/index.html'):
        failures.append('dashboard_integrity_or_permissions')
    models = model_files(STATE, PROJECT, hash_models)
    if not models['vision_cli_exists'] or not models['text_cli_exists'] or any(
            not r['private'] or not r['size_matches'] or r.get('sha256_matches') is False for r in models['files']):
        failures.append('model_installation_or_integrity')
    signatures = {app: command(['/usr/bin/codesign', '--verify', '--deep', '--strict',
                               '/Applications/' + app]).returncode == 0
                  for app in ('PhoneActivityReader.app', 'YourTimeWindowReader.app')}
    if not all(signatures.values()):
        failures.append('reader_signature_invalid')
    network_blocked = probe_network_block(PROJECT)
    filevault = 'FileVault is On' in command(['/usr/bin/fdesetup', 'status']).stdout
    if not network_blocked:
        failures.append('network_sandbox_failed')
    if not filevault:
        failures.append('filevault_not_enabled')
    power = command(['/usr/bin/pmset', '-g', 'batt']).stdout
    memory = command(['/usr/bin/memory_pressure', '-Q']).stdout
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', memory)
    memory_percent = int(match.group(1)) if match else None
    resources = {'ac_power': 'AC Power' in power.splitlines()[0] if power.splitlines() else False,
                 'disk_free_gib': round(shutil.disk_usage(STATE).free / 1024**3, 2),
                 'memory_free_percent': memory_percent, 'load_1m': round(os.getloadavg()[0], 2),
                 'logical_cpus': os.cpu_count()}
    gates = []
    if not resources['ac_power']: gates.append('battery_power')
    if resources['disk_free_gib'] < 10.5: gates.append('disk_below_10_5_gib')
    if memory_percent is None or memory_percent < 20: gates.append('low_memory')
    if resources['load_1m'] > 2 * (resources['logical_cpus'] or 1): gates.append('system_load_high')
    latest = STATE / 'vision-fallback-latest-receipt.json'
    vision = json.loads(latest.read_text()) if latest.is_file() else {}
    return {'schema_version': 1, 'checked_at_utc': now.isoformat(),
            'status': 'needs_repair' if failures else 'ready_with_gates_or_warnings' if gates or warnings else 'ready',
            'failures': failures, 'warnings': warnings, 'jobs': jobs, 'receipts': receipts,
            'database': database, 'private_directories': directories, 'dashboard_integrity': dashboard_ok,
            'models': models, 'reader_signatures': signatures, 'network_blocked': network_blocked,
            'filevault_enabled': filevault, 'resources': resources,
            'overnight': {'start': START.isoformat(timespec='minutes'),
                          'vision_end': VISION_END.isoformat(timespec='minutes'),
                          'end': END.isoformat(timespec='minutes'), 'current_resource_gates': gates,
                          'latest_9b_attempt': {k: vision.get(k) for k in
                                               ('started_at_utc', 'finished_at_utc', 'selected',
                                                'completed', 'failed', 'stop_reason')}},
            'scope': 'Read-only installed configuration, integrity and aggregate receipts. Loaded schedules are not successful runs. No activity bodies or screenshots.'}


def write_private(report: dict, path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix='.setup-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-models', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    report = inspect(hash_models=args.hash_models)
    if args.write:
        write_private(report, STATE / 'setup-latest-receipt.json')
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(1 if report['failures'] else 0)


if __name__ == '__main__':
    main()
