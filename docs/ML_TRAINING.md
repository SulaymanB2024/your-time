# Local specialization training

`specialization_training.py` supplies `dry-run`, `verify`, `train`, and
`validate` stages for the pinned local Qwen3.5 experiment. Importing it or
running `dry-run` does not read exports, import MLX, or load a model.

## Fixed experiment contract

- Model: `mlx-community/Qwen3.5-9B-MLX-4bit`, revision
  `938d8919941c6e7efd3c7150eff7fe9d12afa631`.
- Runtime: MLX-VLM 0.7.6, MLX and MLX Metal 0.32.3, Transformers 5.18.0,
  NumPy 2.4.6, Torch 2.12.1, Torchvision 0.27.1, Pillow 12.3.0 and
  Tokenizers 0.23.2; the complete `requirements-ml.txt` is also fingerprinted.
- QLoRA: rank 8, alpha 16, scale 2, dropout 0, Adam learning rate `1e-5`,
  batch size 1, accumulation 8, sequence limit 2,048, at most two epochs.
- Targets: `gate_proj`, `up_proj`, and `down_proj` in all 32 language MLPs;
  `q_proj`, `k_proj`, `v_proj`, and `o_proj` in the eight full-attention layers
  (indices 3, 7, ..., 31). Exactly 128 modules and 256 adapter arrays train.
  Base, vision, projector, and recurrent projections remain frozen.
- Default seed: 20261005. Default image budget: 512 feature tokens; 1,024 is
  an explicit alternative requiring a separate experiment directory.
- The inference prompt disables thinking. The separately tokenized assistant
  answer and its `<|im_end|>` receive loss; prompt, image and padding tokens
  receive no loss. Oversized examples fail without truncation.

The supplied train and validation JSONL files must contain `id`, one local
`images` path, exactly one user/assistant message pair, and `context`. The user
message must match `activity_context.prompt(context)` and the assistant result
must pass its evidence validation. Optional split, source class,
image hash and episode/group fields are checked when present. Validation
requires 20 real examples; training needs at least two synthetic controls.
Cross-split identity, image hash, episode and near-duplicate group overlaps
are rejected. The harness has no sealed-test interface.

## Controller interface

Run through the existing `.ml-venv` interpreter. All heavy-stage paths are
explicit absolute paths; `--run-dir` must be below the state directory's
`specialization/` folder. The model manifest must be outside the model folder.
For example, with `YOUR_TIME_STATE_DIR` set by the controller:

```sh
.ml-venv/bin/python -B specialization_training.py dry-run

.ml-venv/bin/python -B specialization_training.py verify \
  --model-path "$YOUR_TIME_STATE_DIR/models/qwen3.5-9b-mlx-4bit" \
  --model-manifest "$YOUR_TIME_STATE_DIR/specialization/mlx-asset-manifest.json" \
  --train-jsonl "$YOUR_TIME_STATE_DIR/specialization/export/train.jsonl" \
  --validation-jsonl "$YOUR_TIME_STATE_DIR/specialization/export/validation.jsonl" \
  --run-dir "$YOUR_TIME_STATE_DIR/specialization/training-run" \
  --state-dir "$YOUR_TIME_STATE_DIR" \
  --image-tokens 512 --epochs 2 --max-seconds 5400
```

Use the same paths and recipe for `train` and `validate`. These are bounded
one-shot stages: there is no scheduler or watcher here. Heavy work is admitted
only between 00:30 and 07:00 America/Chicago, leaving a 60-second margin.
The parent uses `model_execution.run_model` with the shared model lock,
`network-off.sb`, and offline environment variables. The worker requires the
inherited locked descriptor. A kernel SIGALRM deadline also terminates a
stalled native call if the parent disappears. The existing resource gate is
checked before launch and periodically during work.

The requested MLX memory limit defaults to 10 GiB (8–12 configurable), with a
256 MiB cache limit. MLX's memory limit is an allocation guideline, not a hard
process-RSS cap. Generation's automatic wired-limit expansion is disabled
inside this worker. Actual device pressure and runtime remain to be measured.

## Smoke checks and checkpoint selection

Every heavy invocation first prepares all train/validation examples to check
image grids, feature-token counts, masks and sequence lengths. Synthetic
controls then exercise image-sensitive logits, zero-adapter parity, finite
nonzero gradients, one update after eight microbatches, unchanged frozen
weights, and adapter save/reload parity. The probe restores the initial
adapter, optimizer and RNG before the requested stage proceeds.

Training generates all 20 validation results for the base and after each
completed epoch. Generations are transient; private state stores aggregate
schema validity, completion-claim flags, activity and candidate correctness,
claim-evidence agreement, affirmative-candidate precision counts and
abstention counts. These checks do not establish semantic entailment of
`visible_work`.

A candidate must preserve baseline schema validity and must not increase
completion-claim flags. Eligible candidates are ranked by fewer flags,
schema validity, joint candidate correctness, activity correctness, individual
candidate correctness, then claim-evidence agreement. Ties retain the prior
best; the base can remain selected. `validate` reloads the selection, repeats
the validation generations, requires matching aggregate scores, and checks
the frozen-weight hash.

## Resume and adapter handoff

The private run directory holds `experiment.json`, `verification.json`,
`progress.json`, and `receipt.json`. Each committed update has a unique
`step-NNNNNN-XXXXXXXX/` directory containing:

```text
adapter/adapter_config.json
adapter/adapters.safetensors
adapter-manifest.json
optimizer.safetensors
progress.json
```

The adapter manifest stays outside the two-file runtime adapter folder. It
binds `base_model_sha256` to `specialization_worker.snapshot_identity` of the
base manifest's file records and includes adapter file sizes and SHA-256s.
The run's `progress.json` points to the last committed checkpoint and carries
`best.checkpoint`; a null best checkpoint means use the base. The controller
can pass `best.checkpoint/adapter` plus its sibling `adapter-manifest.json` to
the worker.

Checkpoint files are closed and fsynced before the run pointer advances.
Adapter snapshots retain history. Generated optimizer files are retained for the
current, immediately preceding and currently selected best checkpoint; older
optimizer files are removed after the new pointer commits. Older adapters remain
available for evaluation but are not general resume points.
Resume restores optimizer/RNG and accepts only the same source, requirements,
model, recipe, and export fingerprints. An interrupted accumulation group is
replayed from the last committed update. Re-run `train` with identical
arguments after a `partial` receipt. Changing the experiment needs a new run
directory. A successful `train` receipt alone is not evidence of improved
sealed-test quality.

## Source verification boundary

The focused tests use synthetic records, fake engines and a NumPy loss mock.
They cover masking, exact targets, leakage rejection, manifest/base binding,
checkpoint selection and recovery, sandbox/lock admission, daytime refusal,
and the native-call hard deadline. They do not load the real model or read
private exports. Actual processor compatibility, device performance,
gradients, full frozen hashes and adapter reload parity must pass `verify`
in the authorized overnight run before training results are accepted.

## Preflight refinements

The assistant positions are gathered before FP32 cross entropy. A tiny CPU
comparison showed identical loss and gradients to the full masked objective;
the frozen vocabulary head still emits full-sequence logits, whose actual peak
memory must be measured. Native MLX RNG is restored using the saved two-word
key through `random.seed`, with exact-key verification; assigning `random.state`
does not restore MLX 0.32.3's native generator.

Each validation generation records a private content-free attempt and decoding
result, with checkpoint array identity and available engine/allocator metrics.
Generated optimizer state retains current, previous and selected best checkpoints
only after committing the new resume pointer; all adapters and progress metadata
remain available. This bounds experiment disk use without erasing caption or
training history. Real image-conditioned gradients, reload integrity and adapter
gains remain acceptance checks for the overnight experiment.

Literal-match validation provides checkpoint preselection only. If it selects the
base, reload validation also checks the final trained checkpoint, and the paired
benchmark evaluates that real adapter instead of omitting the experiment. Its
selection rule is recorded; root semantic review determines acceptance. Further
checkpoint comparisons must occur on validation before the test candidate freeze.
