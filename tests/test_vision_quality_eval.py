import os
import sqlite3
from pathlib import Path

from PIL import Image

import vision_quality_eval as quality


def test_private_review_escapes_untrusted_model_markup():
    data = {"images": [{"path": "/private/example.webp", "timestamp_utc": "2026-09-29T00:00:00+00:00",
                        "models": {"2b_q4": {"description": "<img src=x> [click](https://example.com)"}}}]}
    review = quality.render_review(data)
    assert "<img" not in review
    assert "\\[click\\]" in review
    assert "![Private screenshot](</private/example.webp>)" in review


def test_quality_eval_requires_completed_nonsensitive_ocr(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE screenshots(path TEXT,active_app TEXT,active_window TEXT,ocr_text TEXT,ocr_status TEXT)")
    connection.executemany("INSERT INTO screenshots VALUES (?,?,?,?,?)", [
        ("/one.webp", "Editor", "Draft", "Ordinary visible text", "complete"),
        ("/two.webp", "Editor", "Login", "Enter password", "complete"),
        ("/three.webp", "Editor", "Draft", None, "pending"),
    ])
    connection.commit()
    connection.close()
    monkeypatch.setattr(quality, "DB_PATH", db)
    assert quality.current_ocr(Path("/one.webp")) == "Ordinary visible text"
    assert quality.current_ocr(Path("/two.webp")) is None
    assert quality.current_ocr(Path("/three.webp")) is None


def test_quality_result_write_is_private(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "EVAL_DIR", tmp_path)
    path = tmp_path / "result.json"
    quality.write_private(path, "{}\n")
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_three_model_review_embeds_only_local_images_and_escapes_captions(tmp_path):
    image = tmp_path / "real.webp"
    Image.new("RGB", (8, 8), "blue").save(image)
    data = {"images": [{"path": str(image), "timestamp_utc": "2026-09-29T00:00:00+00:00",
                        "sha256": "sample", "models": {
                            "2b_q4": {"description": "first"},
                            "9b_q8": {"description": "second"},
                            "27b_q3": {"description": "<script>alert('bad')</script>"},
                        }}]}
    page = quality.render_html_review(data, labels=("2b_q4", "9b_q8", "27b_q3"), limit=1)
    assert page.count("data:image/webp;base64,") == 1
    assert page.count("<h3>Caption ") == 3
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "connect-src 'none'" in page
