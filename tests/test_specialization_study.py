from datetime import datetime, timezone

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
