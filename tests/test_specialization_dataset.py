import json

import pytest
from PIL import Image

import specialization_dataset as dataset
from activity_context import CLAIMS, RESULT_VERSION, VERSION, validate
from private_io import atomic_write, write_json
from vision_batch import sha256_file


def example(tmp_path, split):
    identity = "example-" + split
    path = tmp_path / (split + ".png")
    Image.new("RGB", (16, 16), "white").save(path)
    path.chmod(0o600)
    context = {"version": VERSION, "evidence": [{"id": identity, "source": "screen_context"}]}
    annotation = dict(version=RESULT_VERSION, activity_kind="writing", project_candidate=None,
                      task_candidate="Drafting a document", visible_work="A document is visible.",
                      evidence_ids=[identity], uncertainty="supported")
    annotation["claim_evidence"] = {key: [] if annotation[key] is None else [identity] for key in CLAIMS}
    return dict(id=identity, path=str(path), image_sha256=sha256_file(path), context=context,
                annotation=annotation, annotation_provenance={"kind": "test"}, split=split)


def test_export_seals_test_answers_and_keeps_synthetic_scores_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    rows = [example(tmp_path, split) for split in dataset.QUOTAS]
    write_json(tmp_path / "reference-manifest.json", {"review_limit": 100, "examples": rows})
    monkeypatch.setattr(dataset, "synthetic_examples", lambda: [])
    receipt = dataset.export()
    assert receipt["splits"]["train"]["count"] == 1
    assert not (tmp_path / "export/test.jsonl").exists()
    test = (tmp_path / "sealed/test.jsonl").read_text()
    train = (tmp_path / "export/train.jsonl").read_text()
    assert "example-test" in test and "example-test" not in train
    assert json.loads(train)["split"] == "train"
    assert (tmp_path / "sealed/test.jsonl").stat().st_mode & 0o777 == 0o600


def test_export_rejects_changed_images_and_missing_review(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    row = example(tmp_path, "train")
    write_json(tmp_path / "reference-manifest.json", {"review_limit": 100, "examples": [row]})
    monkeypatch.setattr(dataset, "synthetic_examples", lambda: [])
    atomic_write(tmp_path / "train.png", b"changed")
    with pytest.raises(ValueError, match="image changed"):
        dataset.export()
    row["annotation"] = None
    write_json(tmp_path / "reference-manifest.json", {"review_limit": 100, "examples": [row]})
    with pytest.raises(ValueError, match="review is incomplete"):
        dataset.export()


def test_review_quota_and_page_bounds_are_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    write_json(tmp_path / "reference-manifest.json", {"review_limit": 100, "examples": [{"id": str(i)} for i in range(101)]})
    with pytest.raises(ValueError, match="quota exceeded"):
        dataset.read_manifest()
    with pytest.raises(ValueError, match="review page"):
        dataset.contact_sheet("train", -1)


def test_synthetic_examples_have_known_valid_labels_and_private_images(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    rows = dataset.synthetic_examples()
    assert len(rows) == 60
    assert all(r["split"] == "train" for r in rows)
    for row in rows:
        validate(row["annotation"], row["context"])
        assert row["annotation_provenance"]["kind"] == "controlled_synthetic_render"
        assert __import__("pathlib").Path(row["path"]).stat().st_mode & 0o777 == 0o600


def test_contiguous_episode_crosses_midnight_but_splits_on_gap_or_task():
    rows = [dict(path="a", timestamp_utc="2026-10-03T23:59:30+00:00", app="Editor", window="A"),
            dict(path="b", timestamp_utc="2026-10-04T00:00:30+00:00", app="Editor", window="A"),
            dict(path="c", timestamp_utc="2026-10-04T00:01:30+00:00", app="Editor", window="B"),
            dict(path="d", timestamp_utc="2026-10-04T00:04:30+00:00", app="Editor", window="B")]
    identities = dataset.episode_identities(rows)
    assert identities['a'] == identities['b']
    assert identities['b'] != identities['c'] != identities['d']
