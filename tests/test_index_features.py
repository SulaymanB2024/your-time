import json
from datetime import datetime, timezone

import Vision
from Foundation import NSURL
from PIL import Image, ImageDraw

import index_features
import secure_store


def diagnose_synthetic_feature_api(image, tmp_path):
    """Bounded platform diagnostics containing synthetic-image counters only."""
    results = []
    for format_name in ("png", "webp"):
        path = tmp_path / ("diagnostic." + format_name)
        image.save(path)
        for revision in (1, 2):
            for cpu_only in (False, True):
                request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
                request.setRevision_(revision)
                request.setUsesCPUOnly_(cpu_only)
                handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
                    NSURL.fileURLWithPath_(str(path)), {})
                success, error = handler.performRequests_error_([request], None)
                observations = request.results() or []
                results.append({"format": format_name, "revision": revision,
                                "cpu_only": cpu_only, "success": bool(success),
                                "error_code": int(error.code()) if error else None,
                                "elements": [int(item.elementCount()) for item in observations]})
    print(json.dumps({"synthetic_vision_diagnostic": results}))


def test_private_feature_print_index_and_distance(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "activity-ledger.sqlite3")
    monkeypatch.setattr(index_features, "STATE_DIR", tmp_path)
    monkeypatch.setattr(index_features, "DB_PATH", tmp_path / "activity-ledger.sqlite3")
    monkeypatch.setattr(index_features, "SCREENSHOT_ROOTS", (tmp_path,))
    monkeypatch.setattr(index_features, "STATUS_PATH", tmp_path / "feature-receipt.json")
    monkeypatch.setattr(index_features, "resource_gate", lambda benchmark_now: None)
    path = tmp_path / "frame.webp"
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 90, 90), fill="black")
    image.save(path)
    # Surface a safe framework code instead of hiding the integration failure
    # behind the worker's aggregate failure count.
    try:
        original, count = index_features.featureprint(path)
    except ValueError:
        diagnose_synthetic_feature_api(image, tmp_path)
        raise
    assert count == 768
    now = datetime.now(timezone.utc).isoformat()
    with secure_store.connect() as database:
        database.execute(
            "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(path), now, "test", "test", None, "test", "complete", now))
    result = index_features.run(1, allow_battery=True)
    assert result["completed"] == 1 and result["pending_after"] == 0
    with secure_store.connect() as database:
        vector, revision = database.execute(
            "SELECT vector,feature_revision FROM screen_features WHERE path=?", (str(path),)
        ).fetchone()
    assert revision == index_features.REVISION
    assert len(vector) == 3072
    assert index_features.distance(vector, vector) == 0
    assert index_features.distance(original, vector) < 1e-6
    changed = tmp_path / "changed.webp"
    different = Image.new("RGB", (128, 128), "black")
    ImageDraw.Draw(different).ellipse((5, 5, 110, 110), fill="white")
    different.save(changed)
    changed_vector, _ = index_features.featureprint(changed)
    assert index_features.distance(vector, changed_vector) > 0.01
    index_features.write_receipt(result)
    receipt = tmp_path / "feature-receipt.json"
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert "frame.webp" not in receipt.read_text()
    assert json.loads(receipt.read_text())["completed"] == 1
