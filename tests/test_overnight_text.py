import fcntl
import json
import os
import subprocess

import overnight_text


def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight_text, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(overnight_text, 'prepare_private_dir', lambda: None)


def test_receipts_exclude_bodies_and_unstructured_errors():
    output = '\n'.join(['PRIVATE ERROR DETAILS', json.dumps({
        'status': 'complete', 'completed_this_run': 7, 'title': 'PRIVATE TITLE',
        'description': 'PRIVATE CAPTION', 'reason': 'PRIVATE DETAILS', 'selected': 'PRIVATE'}),
        json.dumps({'status': 'resource_gate', 'reason': 'battery_power'})])
    assert overnight_text.output_summaries(output) == [
        {'status': 'complete', 'completed_this_run': 7},
        {'status': 'resource_gate', 'reason': 'battery_power'}]


def test_failed_stage_does_not_hide_receipt_or_prevent_later_stages(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(overnight_text, 'text_budget', lambda _: 1000)
    calls = []
    def worker(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args, 2, '', 'PRIVATE ERROR DETAILS')
        if args[1].endswith('screen_similarity.py'):
            return subprocess.CompletedProcess(args, 0, '{"propagated_count": 2}\n', '')
        return subprocess.CompletedProcess(args, 0, '{"status":"complete","completed_this_run":3}\n', '')
    monkeypatch.setattr(overnight_text.subprocess, 'run', worker)
    result = overnight_text.run()
    assert len(calls) == 5 and result['status'] == 'partial'
    assert result['stages'][0]['status'] == 'failed'
    assert all(r['status'] == 'complete' for r in result['stages'][1:])
    latest = tmp_path/'text-nightly-latest-receipt.json'
    assert json.loads(latest.read_text()) == result
    assert latest.stat().st_mode & 0o777 == 0o600
    history = list((tmp_path/'text-run-receipts').glob('*.json'))
    assert len(history) == 1 and json.loads(history[0].read_text()) == result
    assert 'PRIVATE' not in latest.read_text()


def test_no_model_or_other_worker_starts_outside_text_window(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(overnight_text, 'text_budget', lambda _: 0)
    def unexpected(*args, **kwargs):
        raise AssertionError('Outside-window process launch')
    monkeypatch.setattr(overnight_text.subprocess, 'run', unexpected)
    result = overnight_text.run()
    assert result['status'] == 'skipped' and not result['stages']
    assert result['stop_reason'] == 'outside_text_window'


def test_resource_gate_exit_zero_is_not_reported_as_completion(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(overnight_text, 'text_budget', lambda _: 1000)
    monkeypatch.setattr(overnight_text.subprocess, 'run', lambda args, **kwargs:
        subprocess.CompletedProcess(args, 0, '{"status":"resource_gate","reason":"battery_power"}', ''))
    result = overnight_text.run()
    assert result['status'] == 'partial'
    assert all(r['status'] == 'partial' for r in result['stages'])


def test_second_controller_cannot_overwrite_active_receipt(tmp_path, monkeypatch, capsys):
    isolate(tmp_path, monkeypatch)
    fd = os.open(tmp_path/'text-nightly-controller.lock', os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def unexpected():
            raise AssertionError('Second controller must not run stages')
        monkeypatch.setattr(overnight_text, 'run', unexpected)
        overnight_text.main()
        assert 'another_text_controller_is_running' in capsys.readouterr().out
        assert not (tmp_path/'text-nightly-latest-receipt.json').exists()
    finally:
        os.close(fd)
