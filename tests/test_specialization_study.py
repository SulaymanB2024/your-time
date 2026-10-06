from datetime import datetime, timezone

import pytest

import specialization_study as study
from private_io import write_json


def setup(tmp_path, monkeypatch):
    data = tmp_path / "data"
    root = data / "study"
    monkeypatch.setattr(study, "STUDY_ROOT", root)
    monkeypatch.setattr(study, "DATA_ROOT", data)
    monkeypatch.setattr(study, "CONFIG", root / "config.json")
    monkeypatch.setattr(study, "LATEST", root / "latest.json")
    monkeypatch.setattr(study, "ASSET_MANIFEST", data / "asset.json")
    monkeypatch.setattr(study, "CODE", ())
    write_json(data / "asset.json", {"files": []})
    for split in ("train", "validation"):
        write_json(data / "export" / (split + ".jsonl"), {})
    config = {"source_sha256": {}, "assets_sha256": study.digest(data / "asset.json"),
              "exports": {s: {"sha256": study.digest(data / "export" / (s + ".jsonl"))} for s in ("train", "validation")},
              "configuration_sha256": "a" * 64}
    config["configuration_sha256"] = study.configuration_identity(config)
    write_json(study.CONFIG, config)
    write_json(study.LATEST, {"stage": "verify", "status": "prepared", "nights": [],
                              "configuration_sha256": config["configuration_sha256"]})
    return config


def test_daytime_never_launches_training(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "remaining_seconds", lambda *a, **k: 0)
    monkeypatch.setattr(study.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no child")))
    assert study.run()["status"] == "outside_study_window"


@pytest.mark.parametrize("required,next_stage", [("quality_review", "tune"),
                                                ("tuning_quality_review", "confirm"),
                                                ("confirmation_quality_review", "candidate_quality_review")])
def test_tuning_review_actions_require_exact_stage(tmp_path, monkeypatch, required, next_stage):
    config = setup(tmp_path, monkeypatch)
    calls = []
    with pytest.raises(ValueError, match="tuning_review_stage_required"):
        study.tuning_action(required, next_stage, lambda c: calls.append(c))
    assert not calls
    write_json(study.LATEST, {"stage": required, "status": "awaiting_review", "nights": [],
                             "configuration_sha256": config["configuration_sha256"]})
    study.tuning_action(required, next_stage, lambda c: calls.append(c) or {"status": "reviewed"})
    state = study.read_json(study.LATEST)
    assert state["stage"] == next_stage and calls == [config]


def test_changed_data_or_source_cannot_resume_cached_results(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    write_json(study.DATA_ROOT / "export/train.jsonl", {"changed": True})
    assert study.run()["status"] == "dataset_changed"


def test_used_night_budget_and_review_boundary_stop_without_model(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "remaining_seconds", lambda *a, **k: 20000)
    monkeypatch.setattr(study, "resource_gate", lambda **k: None)
    today = datetime.now(timezone.utc).astimezone(study.ZONE).date().isoformat()
    write_json(study.LATEST, {"stage": "train", "status": "partial", "configuration_sha256": config["configuration_sha256"],
                             "nights": [{"day": today, "elapsed_seconds": study.MAX_NIGHT_SECONDS}]})
    assert study.run()["status"] == "night_study_budget_used"
    write_json(study.LATEST, {"stage": "quality_review", "status": "awaiting_quality_review", "nights": [], "configuration_sha256": config["configuration_sha256"]})
    assert study.run()["status"] == "awaiting_quality_review"


def test_crashed_controller_cannot_receive_another_full_night_budget(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'remaining_seconds', lambda *a, **k: 20000)
    monkeypatch.setattr(study, 'resource_gate', lambda **k: None)
    today = datetime.now(timezone.utc).astimezone(study.ZONE).date().isoformat()
    write_json(study.LATEST, {'stage': 'train', 'status': 'running', 'configuration_sha256': config['configuration_sha256'],
                             'nights': [{'day': today, 'elapsed_seconds': 100,
                                         'reserved_seconds': study.MAX_NIGHT_SECONDS-100}]})
    monkeypatch.setattr(study.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no child')))
    assert study.run()['status'] == 'night_study_budget_used'


def test_delayed_contender_reads_state_again_under_lock(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'remaining_seconds', lambda *a, **k: 20000)
    monkeypatch.setattr(study, 'resource_gate', lambda **k: None)
    original = study.open_private_file
    def concurrent_finish(path, *args, **kwargs):
        write_json(study.LATEST, {'stage': 'quality_review', 'status': 'awaiting_quality_review', 'nights': [], 'configuration_sha256': config['configuration_sha256']})
        return original(path, *args, **kwargs)
    monkeypatch.setattr(study, 'open_private_file', concurrent_finish)
    monkeypatch.setattr(study.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no child')))
    assert study.run() == {'stage': 'quality_review', 'status': 'awaiting_quality_review'}


def test_modified_configuration_is_rejected_before_backend(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    config['thinking'] = True
    write_json(study.CONFIG, config)
    monkeypatch.setattr(study.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no child')))
    assert study.run()['status'] == 'configuration_changed'


def test_held_out_acceptance_requires_review_stage_and_unchanged_source(tmp_path, monkeypatch):
    import specialization_trial

    config = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(specialization_trial, "freeze_trial", lambda *a, **k: pytest.fail("premature freeze"))
    with pytest.raises(ValueError, match="test_review_stage_required"):
        study.accept_test("b" * 64)
    source = tmp_path / "source.py"
    source.write_text("initial")
    config["source_sha256"] = {str(source): study.digest(source)}
    config["configuration_sha256"] = study.configuration_identity(config)
    write_json(study.CONFIG, config)
    write_json(study.LATEST, {"stage": "test_quality_review", "status": "awaiting_test_quality_review",
                            "nights": [], "configuration_sha256": config["configuration_sha256"]})
    source.write_text("changed")
    with pytest.raises(ValueError, match="study_source_changed"):
        study.accept_test("b" * 64)


def test_held_out_acceptance_starts_only_shadow_trial(tmp_path, monkeypatch):
    import specialization_trial

    config = setup(tmp_path, monkeypatch)
    write_json(study.LATEST, {"stage": "test_quality_review", "status": "awaiting_test_quality_review",
                            "nights": [], "configuration_sha256": config["configuration_sha256"]})
    def freeze(received, root, *, reviewed_test_sha256):
        assert received == config and root == study.STUDY_ROOT
        assert reviewed_test_sha256 == "b" * 64
        return {"status": "trial_frozen", "production_changed": False}
    monkeypatch.setattr(specialization_trial, "freeze_trial", freeze)
    assert study.accept_test("b" * 64)["production_changed"] is False
    assert study.read_json(study.LATEST)["stage"] == "trial"


def test_refreeze_only_amends_source_of_unstarted_study_and_archives_prior(tmp_path, monkeypatch):
    config = setup(tmp_path, monkeypatch)
    source = tmp_path / "source.py"
    source.write_text("reviewed source")
    monkeypatch.setattr(study, "CODE", (str(source),))
    result = study.refreeze_unstarted()
    assert result["status"] == "refrozen_unstarted" and result["nights_started"] == 0
    changed = study.read_json(study.CONFIG)
    assert changed["exports"] == config["exports"] and changed["assets_sha256"] == config["assets_sha256"]
    assert study.configuration_matches(changed, study.read_json(study.LATEST))
    assert study.read_json(study.STUDY_ROOT / "configuration-history" / (config["configuration_sha256"] + ".json"))["configuration"] == config


@pytest.mark.parametrize("executed", ["night", "artifact", "dataset"])
def test_refreeze_refuses_executed_or_changed_data_without_mutation(tmp_path, monkeypatch, executed):
    setup(tmp_path, monkeypatch)
    if executed == "night":
        state = study.read_json(study.LATEST)
        state["nights"] = [{"day": "2026-10-05", "elapsed_seconds": 1}]
        write_json(study.LATEST, state)
    elif executed == "artifact":
        write_json(study.STUDY_ROOT / "training/progress.json", {"step": 1})
    else:
        write_json(study.DATA_ROOT / "export/train.jsonl", {"changed": True})
    before = study.CONFIG.read_bytes()
    with pytest.raises(ValueError):
        study.refreeze_unstarted()
    assert study.CONFIG.read_bytes() == before
    assert not (study.STUDY_ROOT / "configuration-history").exists()
