import hashlib
import json
import os
import plistlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone

import setup_check


def test_receipts_exclude_private_bodies_and_use_internal_timestamp(tmp_path):
    now = datetime.now(timezone.utc)
    path = tmp_path / 'reader.json'
    path.write_text(json.dumps({'sampled_at_utc': (now-timedelta(minutes=4)).isoformat(),
                               'trusted': True, 'title': 'PRIVATE WINDOW',
                               'ocr_text': 'PRIVATE OCR', 'status': 'sampling',
                               'stop_reason': 'PRIVATE ERROR BODY'}))
    path.chmod(0o600)
    result = setup_check.receipt_summary(path, now, 30)
    assert result['fresh'] is False  # Touching a stale receipt does not make it fresh.
    assert result['private'] is True
    assert result['fields'] == {'trusted': True, 'status': 'sampling'}
    assert 'PRIVATE' not in json.dumps(result)


def test_job_loaded_is_distinct_from_running_and_source_drift(tmp_path, monkeypatch):
    source, installed = tmp_path/'source.plist', tmp_path/'installed.plist'
    source.write_bytes(plistlib.dumps({'Label': 'example', 'ProgramArguments': ['/bin/zsh', '/task/job.zsh'],
                                      'StartCalendarInterval': {'Hour': 0, 'Minute': 30}}))
    installed.write_bytes(source.read_bytes())
    installed.chmod(0o600)
    monkeypatch.setattr(setup_check, 'command', lambda args: subprocess.CompletedProcess(
        args, 0, 'state = not running\narguments = /bin/zsh /task/job.zsh\n', ''))
    result = setup_check.inspect_job(source, installed)
    assert result['loaded'] and result['scheduled'] and result['arguments_loaded']
    assert result['state'] == 'not running'
    assert result['last_exit_code'] is None
    assert result['matches_source']
    installed.write_bytes(plistlib.dumps({'Label': 'example', 'ProgramArguments': ['/task/other']}))
    assert not setup_check.inspect_job(source, installed)['matches_source']


def test_hash_verification_rejects_changed_weights_without_inference(tmp_path):
    folder = tmp_path/'models/qwen3.5-9b-q8-eval'
    folder.mkdir(parents=True)
    files = []
    for name in ['Qwen3.5-9B-Q8_0.gguf', 'mmproj-F16.gguf']:
        p = folder/name
        p.write_bytes(b'fixture')
        p.chmod(0o400)
        files.append({'name': name, 'size': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()})
    (tmp_path/'vision_quality_model_manifest.json').write_text(json.dumps({'files': files}))
    result = setup_check.model_files(tmp_path, tmp_path, True)
    assert all(r['sha256_matches'] for r in result['files'])
    p.chmod(0o600)
    p.write_bytes(b'changed')
    result = setup_check.model_files(tmp_path, tmp_path, True)
    assert not result['files'][-1]['sha256_matches']


def test_missing_dashboard_and_escaped_model_path_are_not_ready(tmp_path):
    assert not setup_check.dashboard_health(tmp_path)
    (tmp_path/'vision_quality_model_manifest.json').write_text(json.dumps({'files': [
        {'name': '../private', 'size': 1, 'sha256': 'x'}]}))
    assert setup_check.model_files(tmp_path, tmp_path, False)['manifest_invalid']


def test_receipt_write_is_private_and_atomic(tmp_path):
    path = tmp_path/'setup.json'
    setup_check.write_private({'status': 'ready'}, path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == {'status': 'ready'}
    assert {p.name for p in tmp_path.iterdir()} == {'setup.json'}


def test_malformed_job_is_reported_without_exposing_its_body(tmp_path):
    source = tmp_path/'source.plist'
    source.write_text('PRIVATE INVALID PLIST')
    result = setup_check.inspect_job(source, tmp_path/'installed.plist')
    assert result['source_invalid'] and not result['loaded']
    assert 'PRIVATE' not in json.dumps(result)
    source.write_bytes(plistlib.dumps({'Label': 'example', 'ProgramArguments': '/wrong/type'}))
    assert setup_check.job_definition(source) is None


def test_dashboard_checks_retry_publication_race_but_reject_corruption(tmp_path, monkeypatch):
    folder = tmp_path/'dashboard'
    folder.mkdir()
    page = folder/'index.html'
    page.write_bytes(b'new page')
    page.chmod(0o600)
    (folder/'manifest.json').write_text(json.dumps({'bytes': 8, 'sha256': hashlib.sha256(b'new page').hexdigest()}))
    real_read = Path.read_bytes
    reads = 0
    def race(path):
        nonlocal reads
        if path == page:
            reads += 1
            if reads == 1: return b'previous page'
        return real_read(path)
    monkeypatch.setattr(Path, 'read_bytes', race)
    assert setup_check.dashboard_health(tmp_path) and reads == 2
    page.write_bytes(b'corrupt')
    assert not setup_check.dashboard_health(tmp_path)


def test_invalid_vision_receipt_does_not_crash_or_expose_activity(tmp_path):
    path = tmp_path/'vision.json'
    for body in ('INVALID PRIVATE BODY', '[]', json.dumps({'completed': -1, 'description': 'PRIVATE CAPTION'}),
                 json.dumps({'stop_reason': 'PRIVATE ERROR DETAILS'})):
        path.write_text(body)
        result = setup_check.vision_attempt_summary(path)
        assert result == {'exists': True, 'valid': False}
    path.write_text(json.dumps({'completed': 2, 'stop_reason': 'battery_power', 'description': 'PRIVATE CAPTION'}))
    result = setup_check.vision_attempt_summary(path)
    assert result['valid'] and result['completed'] == 2
    assert 'PRIVATE' not in json.dumps(result)


def test_duplicate_model_files_and_invalid_hashes_fail_closed(tmp_path):
    manifest = tmp_path/'vision_quality_model_manifest.json'
    files = [{'name': name, 'size': 7, 'sha256': '0'*64}
             for name in ('Qwen3.5-9B-Q8_0.gguf', 'mmproj-F16.gguf')]
    manifest.write_text(json.dumps({'files': files + [files[0]]}))
    assert setup_check.model_files(tmp_path, tmp_path, False)['manifest_invalid']
    files[0]['sha256'] = 'invalid'
    manifest.write_text(json.dumps({'files': files}))
    assert setup_check.model_files(tmp_path, tmp_path, False)['manifest_invalid']
