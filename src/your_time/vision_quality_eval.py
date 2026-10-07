"""Compare pinned local vision models on the same private screenshots."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from activity_context import parse_result, raw_safety_checks
from engine_identity import llama_identity
from inference_telemetry import record_result
from model_execution import run_model
from private_io import atomic_write
from secure_store import DB_PATH, STATE_DIR
from vision_batch import (
    LLAMA_CLI,
    PROMPT,
    PROMPT_VERSION,
    SANDBOX_PROFILE,
    SENSITIVE_RE,
    clean_description,
    sha256_file,
)

EVAL_DIR = STATE_DIR / "vision-quality-eval"
SELECTION = EVAL_DIR / "selection.json"
RESULTS = EVAL_DIR / "results.json"
REVIEW = EVAL_DIR / "review.md"
REVIEW_HTML = EVAL_DIR / "review.html"
REVIEW_PEAK_HTML = EVAL_DIR / "review-three-models.html"
MIN_FREE_BYTES = 20 * 1024**3
MODEL_SPECS = {
    "2b_q4": (
        Path(__file__).with_name("vision_model_manifest.json"),
        STATE_DIR / "models/qwen3.5-2b-q4km",
    ),
    "9b_q8": (
        Path(__file__).with_name("vision_quality_model_manifest.json"),
        STATE_DIR / "models/qwen3.5-9b-q8-eval",
    ),
    "27b_q3": (
        Path(__file__).with_name("vision_peak_model_manifest.json"),
        STATE_DIR / "models/qwen3.5-27b-q3-eval",
    ),
}
COMPLETE_ACTION_RE = re.compile(r"\b(sent|submitted|purchased|deleted|saved|clicked|emailed|posted)\b", re.I)
IDENTIFIER_RE = re.compile(r"https?://|[\w.+-]+@[\w.-]+\.\w+|\b\d{4,}\b", re.I)
WORD_RE = re.compile(r"[a-z]{4,}", re.I)


def write_private(path: Path, value: str) -> None:
    atomic_write(path, value.encode("utf-8"))


def on_ac_power() -> bool:
    result = subprocess.run(["/usr/bin/pmset", "-g", "batt"], check=True,
                            capture_output=True, text=True, timeout=10)
    return "AC Power" in result.stdout.splitlines()[0]


def resource_gate(*, allow_battery: bool = False) -> str | None:
    if not allow_battery and not on_ac_power():
        return "battery_power"
    if shutil.disk_usage(STATE_DIR).free < MIN_FREE_BYTES:
        return "low_disk"
    if os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
        return "high_load"
    report = subprocess.run(["/usr/bin/memory_pressure", "-Q"], check=True,
                            capture_output=True, text=True, timeout=10).stdout
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", report)
    if not match or int(match.group(1)) < 20:
        return "low_memory"
    return None


def verify_models(labels: list[str]) -> dict:
    if not LLAMA_CLI.is_file():
        raise RuntimeError("Pinned llama-mtmd-cli is missing")
    models = {}
    for label in labels:
        manifest_path, folder = MODEL_SPECS[label]
        manifest = json.loads(manifest_path.read_text())
        files = {}
        for item in manifest["files"]:
            path = folder / item["name"]
            if not path.is_file() or path.is_symlink() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"Model verification failed: {label} / {item['name']}")
            files[item["name"]] = path
        weights = next(path for name, path in files.items() if name.startswith("Qwen"))
        projector = next(path for name, path in files.items() if name.startswith("mmproj-F16"))
        models[label] = {"weights": weights, "projector": projector,
                         "projector_sha256": next(item["sha256"] for item in manifest["files"] if item["name"] == projector.name),
                         "weights_sha256": next(item["sha256"] for item in manifest["files"] if item["name"] == weights.name)}
    return models


def current_ocr(path: Path) -> str | None:
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT active_app,active_window,ocr_text,ocr_status FROM screenshots WHERE path=?",
            (str(path),),
        ).fetchone()
    finally:
        connection.close()
    if not row or row[3] != "complete" or SENSITIVE_RE.search(" ".join(value or "" for value in row[:3])):
        return None
    return row[2] or ""


def run_image(path: Path, model: dict, *, timeout_seconds: int | None = None,
              max_tokens: int = 1024, prompt: str = PROMPT,
              prompt_version: str = PROMPT_VERSION, variant: str = "production",
              image_side: int = 1600, image_tokens: int = 1024,
              threads: int = 4, context: dict | None = None, experiment_sha256: str | None = None) -> dict:
    if image_side not in {1024, 1600, 2048} or image_tokens not in {512, 1024, 1536} or threads not in {2, 4, 8}:
        raise ValueError("Unsupported benchmark configuration")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="quality-frame-", dir=STATE_DIR) as temp:
        os.chmod(temp, 0o700)
        png = Path(temp) / "frame.png"
        with Image.open(path) as source:
            reduced = source.convert("RGB")
            reduced.thumbnail((image_side, image_side), Image.Resampling.LANCZOS)
            reduced.save(png)
            width, height = reduced.size
        os.chmod(png, 0o600)
        prompt_path = Path(temp) / "prompt.txt"
        fd = os.open(prompt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(prompt)
        prepare_seconds = time.monotonic() - started
        command = [
            "/usr/bin/sandbox-exec", "-f", str(SANDBOX_PROFILE), str(LLAMA_CLI),
            "-m", str(model["weights"]), "--mmproj", str(model["projector"]),
            "--image", str(png), "-f", str(prompt_path), "-c", "4096", "-n", str(max_tokens),
            "-ngl", "99", "-t", str(threads), "-tb", str(threads), "--image-max-tokens", str(image_tokens), "--temp", "0", "--jinja",
        ]
        # The manual quality comparison has no wall-clock cutoff. A scheduled
        # fallback can pass a bound so one frame cannot consume the entire night.
        try:
            completed = run_model(command, state_dir=STATE_DIR, capture_output=True,
                                       timeout=timeout_seconds,
                                       telemetry={"stage": "vision", "variant": variant,
                                                  "model_sha256": model.get("weights_sha256"),
                                                  "experiment_sha256": experiment_sha256,
                                                  "input_sha256": sha256_file(path),
                                                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                                  "projector_sha256": model.get("projector_sha256"),
                                                  "prompt_version": prompt_version,
                                                  "engine_version": "llama.cpp-0.5.0",
                                                  "engine_sha256": llama_identity(LLAMA_CLI),
                                                  "image_prepare_seconds": prepare_seconds,
                                                  "image_width": width, "image_height": height},
                                       env={"HOME": str(Path.home()),
                                            "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                                            "LANG": "en_US.UTF-8"})
        except subprocess.TimeoutExpired as error:
            return {"status": "timeout", "elapsed_seconds": round(time.monotonic() - started, 2),
                    "telemetry_attempt_id": getattr(error, "telemetry_attempt_id", None)}
    if completed.returncode:
        return {"status": "model_error", "exit_code": completed.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "telemetry_attempt_id": getattr(completed, "telemetry_attempt_id", None)}
    safety = raw_safety_checks(completed.stdout)
    if context is not None:
        try:
            activity = parse_result(completed.stdout, context)
            status = "complete"
        except json.JSONDecodeError:
            activity, status = None, "decoding_error"
        except ValueError:
            activity, status = None, "schema_error"
        attempt_id = getattr(completed, "telemetry_attempt_id", None)
        record_result(STATE_DIR, attempt_id, status)
        return {"status": status, "activity": activity,
                "raw_safety_checks": safety,
                "elapsed_seconds": round(time.monotonic() - started, 2), "telemetry_attempt_id": attempt_id}
    caption = clean_description(completed.stdout)
    record_result(STATE_DIR, getattr(completed, "telemetry_attempt_id", None), "complete" if caption else "incomplete")
    return {"status": "complete" if caption else "incomplete",
            "raw_safety_checks": safety,
            "description": caption, "elapsed_seconds": round(time.monotonic() - started, 2),
            "telemetry_attempt_id": getattr(completed, "telemetry_attempt_id", None)}


def proxies(description: str | None, ocr: str) -> dict:
    if not description:
        return {"unclear": True, "completed_action_claim": False,
                "identifier_leak": False, "literal_ocr_word_overlap": 0.0}
    words = set(WORD_RE.findall(description.casefold()))
    evidence = set(WORD_RE.findall(ocr.casefold()))
    return {
        "unclear": "visible task is unclear" in description.casefold(),
        "completed_action_claim": bool(COMPLETE_ACTION_RE.search(description)),
        "identifier_leak": bool(IDENTIFIER_RE.search(description)),
        "literal_ocr_word_overlap": round(len(words & evidence) / len(words), 3) if words else 0.0,
    }


def render_review(data: dict) -> str:
    lines = ["# Private vision comparison", "", "Open this file locally. The captions are untrusted model inferences.",
             "Captions A and B are shown in a varying blind order. Mark each as accurate, partly accurate, or wrong after inspecting its image.", ""]
    for index, item in enumerate(data["images"], 1):
        lines += [f"## Image {index}", "", f"![Private screenshot](<{item['path']}>)", "",
                  f"Capture time: {item['timestamp_utc']}", ""]
        labels = ("2b_q4", "9b_q8")
        if int(hashlib.sha256(item.get("sha256", item["path"]).encode()).hexdigest(), 16) % 2:
            labels = labels[::-1]
        for letter, label in zip("AB", labels):
            result = item.get("models", {}).get(label, {})
            caption = html.escape(result.get("description") or "[" + result.get("status", "pending") + "]")
            caption = re.sub(r"([\\`*_{}\[\]()#+.!|>])", r"\\\1", caption)
            lines += [f"**Caption {letter}:** {caption}", "",
                      "Rating: [ ] accurate  [ ] partly accurate  [ ] wrong  [ ] sensitive detail", ""]
    return "\n".join(lines)


def render_html_review(data: dict, *, labels: tuple[str, ...] = ("2b_q4", "9b_q8"),
                       limit: int | None = None) -> str:
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; img-src data:; style-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'\">",
        "<title>Private vision comparison</title>",
        f"<style>body{{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;background:#111;color:#eee}}section{{border-top:1px solid #777;padding:2rem 0}}img{{max-width:100%;height:auto;border:1px solid #888}} .pair{{display:grid;grid-template-columns:repeat({len(labels)},1fr);gap:1rem}}article{{background:#222;padding:1rem;border-radius:.6rem}}small{{color:#aaa}}</style>",
        "<h1>Private vision comparison</h1><p>All images and captions are embedded in this local file. The page has no scripts or network access. Rate each caption after looking at the screenshot; A/B order varies by image.</p>",
    ]
    for index, item in enumerate(data["images"][:limit], 1):
        path = Path(item["path"])
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        seed = item.get("sha256", item["path"])
        blind_order = sorted(labels, key=lambda label: hashlib.sha256((seed + label).encode()).hexdigest())
        parts.append(f"<section><h2>Image {index}</h2><small>{html.escape(item['timestamp_utc'])}</small><p><img alt='Private screenshot {index}' src='data:image/webp;base64,{encoded}'></p><div class='pair'>")
        for letter, label in zip("ABC", blind_order):
            result = item.get("models", {}).get(label, {})
            caption = html.escape(result.get("description") or "[" + result.get("status", "pending") + "]")
            parts.append(f"<article><h3>Caption {letter}</h3><p>{caption}</p><p>Rating: accurate / partly accurate / wrong / sensitive detail</p></article>")
        parts.append("</div></section>")
    parts.append("</html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected images, then resume later")
    parser.add_argument("--allow-battery", action="store_true", help="Allow this bounded test on battery power")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=["2b_q4", "9b_q8"])
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    os.umask(0o077)
    gate = resource_gate(allow_battery=args.allow_battery)
    if gate:
        raise RuntimeError(f"Quality test blocked by resource gate: {gate}")
    selection_bytes = SELECTION.read_bytes()
    selection = json.loads(selection_bytes)
    models = verify_models(args.models)
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text())
        if data["selection_sha256"] != selection_sha:
            raise RuntimeError("Existing results belong to a different image selection")
        for label, item in models.items():
            if label in data["models"] and data["models"][label] != item["weights_sha256"]:
                raise RuntimeError(f"Existing results use a different {label} model")
            data["models"][label] = item["weights_sha256"]
    else:
        data = {"selection_sha256": selection_sha, "prompt_version": PROMPT_VERSION,
                "models": {label: item["weights_sha256"] for label, item in models.items()},
                "images": [dict(item, models={}) for item in selection["images"]]}
    for index, item in enumerate(data["images"][:args.limit], 1):
        path = Path(item["path"])
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to((STATE_DIR / "screenshots").resolve())
                or path.stat().st_uid != os.getuid() or sha256_file(path) != item["sha256"]):
            raise RuntimeError(f"Screenshot source changed for image {index}")
        ocr = current_ocr(path)
        if ocr is None:
            item["status"] = "sensitive_or_ocr_unavailable"
            continue
        for label, model in models.items():
            if item["models"].get(label, {}).get("status") == "complete":
                continue
            gate = resource_gate(allow_battery=args.allow_battery)
            if gate:
                write_private(RESULTS, json.dumps(data, indent=2) + "\n")
                raise RuntimeError(f"Quality test stopped at resource gate: {gate}")
            result = run_image(path, model)
            result.update(proxies(result.get("description"), ocr))
            item["models"][label] = result
            write_private(RESULTS, json.dumps(data, indent=2) + "\n")
            print(f"image {index}/{len(data['images'])} {label}: {result['status']} in {result['elapsed_seconds']}s", flush=True)
    write_private(REVIEW, render_review(data))
    write_private(REVIEW_HTML, render_html_review(data))
    if "27b_q3" in data["models"]:
        write_private(REVIEW_PEAK_HTML, render_html_review(data, labels=("2b_q4", "9b_q8", "27b_q3"), limit=3))
    summary = {}
    for label in data["models"]:
        runs = [item.get("models", {}).get(label, {}) for item in data["images"]]
        complete = [run for run in runs if run.get("status") == "complete"]
        summary[label] = {
            "complete": len(complete), "failed_or_incomplete": len(runs) - len(complete),
            "median_seconds": round(statistics.median(run["elapsed_seconds"] for run in complete), 1) if complete else None,
            "unclear": sum(run.get("unclear", False) for run in complete),
            "completed_action_claim_flags": sum(run.get("completed_action_claim", False) for run in complete),
            "identifier_leak_flags": sum(run.get("identifier_leak", False) for run in complete),
            "median_literal_ocr_word_overlap": round(statistics.median(run["literal_ocr_word_overlap"] for run in complete), 3) if complete else None,
        }
    write_private(EVAL_DIR / "summary.json", json.dumps({"images": len(data["images"]),
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(), "models": summary,
                   "caveat": "Literal OCR overlap and string flags are weak proxies, not a human accuracy grade."}, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
