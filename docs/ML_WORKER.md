# Resident MLX specialization worker

`specialization_worker.py` is an opt-in Python controller and private stdio
worker for the local Qwen3.5-9B MLX snapshot. Importing the module does not load
a model. It does not change the production Q8 pipeline or install a scheduler.

## Controller interface

The caller prepares the evidence pack with `activity_context` and supplies an
optional approved screenshot path and SHA-256. The worker does not read the
ledger database, assemble evidence, or enumerate screenshots.

```python
from specialization_worker import WorkerConfig, WorkerController

config = WorkerConfig(
    model_dir=str(state_dir / "models/qwen3.5-9b-mlx-4bit"),
    model_manifest=str(state_dir / "specialization/mlx-asset-manifest.json"),
    state_dir=str(state_dir),
    image_roots=(str(approved_image_root),),
    # Optional; supply both together:
    # adapter_dir=str(adapter_dir),
    # adapter_manifest=str(adapter_manifest),
    max_seconds=1800,
    max_requests=100,
    startup_seconds=300,
)

with WorkerController(config) as controller:
    response = controller.request(
        evidence_pack,
        image_path=str(approved_image),
        image_sha256=approved_image_sha256,
        context_tokens=2048,
        image_tokens=512,
        max_tokens=768,
        thinking=False,
        timeout_seconds=180,
    )
    validated_result = response["result"]
```

The response includes `engine_metrics`, `memory_metrics`,
`telemetry_attempt_id`, and `telemetry_available`. `result` passes
`activity_context.parse_result` in both processes, including per-claim evidence
validation. There is no automatic retry; after an error, the controller reaps
its child and closes. A caller may explicitly create another controller within
its own bounded retry policy. Concurrent calls raise `ModelBusy`.

Request defaults are `thinking=False`, `thinking_budget=256`, `max_tokens=768`,
`timeout_seconds=180`, `image_side=1600`, `image_tokens=1024`,
`context_tokens=4096`, and `seed=0`. Accepted context limits are 2,048 and 4,096;
image budgets are 512, 1,024, and 1,536; image-side caps are 1,024, 1,600, and
2,048. Generation is greedy. The prepared prompt's actual token count plus
`max_tokens` must fit the context limit, and expanded image tokens must fit the
image budget. Output is capped at 32 KiB; a token-limit finish is rejected.
Each JSON frame is capped at 64 KiB. Request timeouts are 1–300 seconds.

## Pinned assets and runtime

The worker requires installed `mlx-vlm==0.7.6`, `mlx==0.32.3`,
`mlx-metal==0.32.3`, `transformers==5.18.0`, `numpy==2.4.6`,
`torch==2.12.1`, `torchvision==0.27.1`, `pillow==12.3.0`, and
`tokenizers==0.23.2`. Torch/Torchvision initialize the native processor; MLX
executes model weights. It loads only a local Qwen3.5
configuration with hidden size 4,096 and 32 language layers, using strict base
weights, `trust_remote_code=False`, and `local_files_only=True`.

Each manifest is a private JSON object containing a nonempty `files` array:

```json
{"files": [{"name": "config.json", "size": 123, "sha256": "<64 lowercase hex characters>"}]}
```

Pin every runtime JSON, Jinja, safetensors, tokenizer model/text, and Python
file. Names are unique basenames. Hashes are checked against the full bytes
before loading. `snapshot_identity(files)` returns the canonical snapshot
identity used in receipts. Keep manifests outside model/adapter directories;
the explicitly supplied manifest is exempt from closure checking if inside.
Files must be owned, private, regular, single-link files; model and adapter
directories must also be private. Symlinked paths are rejected.

An adapter manifest additionally contains `base_model_sha256`, equal to
`snapshot_identity(base_manifest["files"])`. An adapter directory contains
`adapter_config.json` and `adapters.safetensors`, both pinned. LoRA config must
provide explicit, unique `language_model.*` target names and rank, scale, and
dropout. Every A/B weight key and shape must exactly match those targets before
layers attach; the attached trainable-key set must also match. Extra/missing
keys and mismatched base identities are errors.

## Isolation and lifetime

The controller holds `STATE_DIR/local-model-execution.lock` for the entire
resident lifetime and passes that same locked descriptor to its worker.
The child retains exclusion if the controller exits. It runs under the
repository's `network-off.sb`, with Hugging Face and Transformers offline
flags, no listener, and stdout reserved for the versioned stdio protocol.
Model diagnostics go to a discarded stderr stream.

Admission checks use the existing 00:30–07:00 America/Chicago vision window,
AC, free memory, disk, and load gates before launch/load and between requests.
Snapshot verification checks deadlines every MiB and resources between files
and at most every five seconds within a file. Generation checks deadlines
after each yielded token and resources at most every five seconds. Loading and
generation run under kernel alarms with default termination, independently of
Python's GIL and the parent. The worker is also bounded by its configured
lifetime and request count. The controller waits for and reaps only its own
child before releasing its descriptor.

Each request creates a fresh prompt cache, including recurrent state, clears
position/rope state and tokenizer thinking state, resets stopping criteria and
RNG, and clears allocator cache before and after generation. No prompt cache,
vision cache, prefix cache, or detokenizer is shared between requests. Input
images are private, within configured roots, SHA-verified in memory, bounded
to 32 MiB/32 million pixels, converted/resized in memory, and never persisted
by the worker.

## Metrics and verification limits

`load_seconds` measures model/adapter load and synchronization, excluding
manifest hashing and imports. A separate startup attempt owns that cost;
request attempts link to it and report zero additional load. Request timings
include time to first token from before the initial state reset, elapsed
generation after the first token, both state reset/cleanup costs, and full
backend request wall time. They do not claim to isolate vision
encoding from text prefill. Prompt and generation token counts come from the
runtime's completion record.

Attempt summaries do not estimate sustainable nightly capacity: startup,
preparation between attempts, worker teardown and interrupted sessions require
the full benchmark/trial wall-time journal. Unknown elapsed costs remain null;
crash reservations are conservative bounds, not measured durations.

`memory_metrics` has `scope="mlx_allocator"` and active/cache/peak byte counts.
These are process allocator measurements, not device-wide memory ownership.
The existing telemetry sampler attempts process CPU/RSS and labels AGX GPU
measurements as whole-device; unavailable samples remain null. Receipts store
numeric measurements and approved identities, never evidence, prompts,
thinking, raw model output, or exception bodies.

`process_observer.py` provides separate supplemental measurements when macOS
refuses the sampler's privileged `ps` executable inside the network sandbox.
It uses native process counters, binds the running overnight job by PID and
process start identity, and visits only owned descendants. CPU percentages are
measured over each interval using the current Mach timebase. RSS and physical
footprint stay per process; shared memory is never summed. These job-process
observations have no per-attempt linkage and never replace historical nulls.

The observer has its own single-instance lock, a private bounded JSONL spool
and a final receipt with the spool hash. It stops when its exact job ends, at
its explicit time/storage limit, or before the vision window closes. A worker
timer bounds its lifetime even if the launching process disappears. The CLI
launches its worker under the existing network-denial profile. It neither takes
the model lock nor loads weights, and it never signals an observed process.
Check actual samples, final status and hashes before accepting its measurements.

Focused verification uses synthetic packs, mocked MLX state, and disposable
stdio subprocesses. It covers request isolation, bounds, hashes, adapter shapes,
private receipts, inherited exclusion, cleanup, and native-call deadline
termination. Installed runtime symbols were checked without loading weights.
Real-model startup, adapter execution, quality, throughput, and memory behavior
remain to be measured in an authorized overnight run.

## Verified preprocessing and telemetry amendments

MLX-VLM 0.7.6 drops pixel-budget call arguments in its native input wrapper.
`configure_pixels` sets the actual processor properties for each request and the
expanded-token guard remains mandatory. `tools/check_ml_runtime.py` verifies
native preprocessing on generated images, RNG replay, and loss-gradient parity
without loading model weights or reading private images.

Each request binds the image/context, decoding and image settings, instruction
prompt, verified chat template, worker source and pinned preprocessing/runtime
versions. Startup owns model-load cost; request receipts link to it and report
zero additional load. Schema failures preserve raw-final safety flags, failed
request receipt IDs survive exceptions, and benchmark failures retain those IDs.
Measured warm speed and real semantic performance still require overnight runs.
