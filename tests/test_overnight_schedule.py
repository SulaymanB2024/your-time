import importlib
import sys
from datetime import datetime, timezone

import pytest

from overnight_schedule import VISION_END, in_window, remaining_seconds, text_budget


def local_utc(hour, minute=0):
    return datetime(2026, 10, 5, hour + 5, minute, tzinfo=timezone.utc)


def test_overall_and_dedicated_windows_have_exact_boundaries():
    assert not in_window(local_utc(0, 29))
    assert in_window(local_utc(0, 30))
    assert in_window(local_utc(7, 59))
    assert not in_window(local_utc(8))
    assert in_window(local_utc(6, 59), end=VISION_END)
    assert not in_window(local_utc(7), end=VISION_END)
    assert remaining_seconds(local_utc(0, 30)) == 27000


def test_text_budget_cannot_start_before_seven_or_run_past_eight():
    assert text_budget(1200, local_utc(6, 59)) == 0
    assert text_budget(1200, local_utc(7)) == 1200
    assert text_budget(1200, local_utc(7, 58)) == 100
    assert text_budget(1200, local_utc(8)) == 0


def test_dst_remaining_time_uses_real_elapsed_time():
    # 00:30 before the fall-back transition has 8.5 real hours until 08:00.
    fall = datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)
    spring = datetime(2026, 3, 8, 6, 30, tzinfo=timezone.utc)
    assert remaining_seconds(fall) == 8.5 * 3600
    assert remaining_seconds(spring) == 6.5 * 3600


@pytest.mark.parametrize('module_name', ['window_topic_tagging', 'local_synthesis', 'screen_context_tagging'])
def test_scheduled_text_workers_do_not_load_models_outside_the_window(module_name, monkeypatch, capsys):
    import overnight_schedule
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, 'argv', [module_name, '--overnight-only'])
    monkeypatch.setattr(overnight_schedule, 'text_budget', lambda *_: 0)
    def unexpected_load():
        raise AssertionError('A scheduled worker attempted to load a model outside its window')
    monkeypatch.setattr(module, 'verify_model', unexpected_load)
    module.main()
    assert 'outside_text_window' in capsys.readouterr().out


def test_resumed_tagger_rechecks_wall_clock_before_inference(tmp_path, monkeypatch):
    from datetime import date

    import overnight_schedule
    import window_topic_tagging as worker
    monkeypatch.setattr(worker, 'ANALYSIS_DIR', tmp_path)
    monkeypatch.setattr(worker, 'source_rows', lambda _: ([
        {'id':'test','title':'Chess tutorial','strong_captions':[], 'seconds':120}], 120))
    monkeypatch.setattr(overnight_schedule, 'text_budget', lambda *_: 0)
    def unexpected_call(*_args):
        raise AssertionError('A resumed worker inferred after its wall-clock window')
    monkeypatch.setattr(worker, 'model_call', unexpected_call)
    report = worker.run_day(date(2026,10,3), tmp_path/'fake.gguf','sha',
                            max_seconds=500, overnight_only=True)
    assert report['stop_reason'] == 'text_window_ended'
    assert report['tagged_titles'] == 0


@pytest.mark.parametrize('stage_index', [0, 1, 2])
def test_exact_scheduled_stage_arguments_are_accepted(stage_index, monkeypatch, capsys):
    import overnight_schedule
    import overnight_text
    _, script, args = overnight_text.STAGES[stage_index]
    module = importlib.import_module(script.removesuffix('.py'))
    monkeypatch.setattr(sys, 'argv', [script, *args])
    monkeypatch.setattr(overnight_schedule, 'text_budget', lambda *_: 0)
    module.main()  # Argument validation happens before the outside-window return.
    assert 'outside_text_window' in capsys.readouterr().out
