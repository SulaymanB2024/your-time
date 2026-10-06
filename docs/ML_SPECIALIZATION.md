# Qwen3.5 specialization

## Acceptance contract

Improve accurate project/task identification and daily temporal coverage using the
existing Qwen3.5-9B baseline. Timings alone do not establish task quality. Recorded
durations remain collector measurements; model outputs remain evidence-linked
inferences. Activity, screenshots, annotations, adapters and telemetry stay in the
private local state directory and never enter Git or hosted CI.

The coordinating chat may review at most 100 sensitive-filtered examples and their
associated context. Specialist chats receive code and aggregate measurements only.
Keep all other apps open. Heavy experiments run 00:30–07:00 America/Chicago; the
07:00–08:00 synthesis reservation remains intact. One model at a time, with the
existing inherited advisory lock and network sandbox.

## Experiments

Compare production Q8, improved prompt/context Q8, unchanged quantized MLX 9B,
and the same MLX model with a learned adapter. Keep preprocessing and held-out
examples identical within paired comparisons. Separate cold startup from warm
request costs. Test thinking modes, image budgets and CPU threads independently.

Reference examples: 60 train / 20 validation / 20 sealed test, with whole episodes
and visually similar frames kept together. Later dates supply held-out evaluation.
Only directly supported labels are references; personal intent is unconfirmed.
Synthetic examples have explicit ground truth and are never included in real-test
accuracy. Source model captions are unverified suggestions, not training answers.

Initial QLoRA: frozen vision tower, language-layer rank 8 / alpha 16, batch 1,
gradient accumulation 8, checkpointing, learning rate 1e-5, sequence length 2048,
assistant-only loss, at most two epochs. Verify image tensors, loss masks, finite
nonzero gradients, adapter reload and memory before training a real experiment.
Select checkpoints with validation only. Seal test annotations from trainers.

Promotion requires held-out gains over improved context/prompt, no observed
sensitive disclosure or unsupported completion claims, and a three-night trial.
Report pilot sample uncertainty. Use successful unique episode coverage, failed
attempts and total elapsed to calculate sustainable capacity with a 20% reserve.

## Overnight study

`specialization_study.py` freezes code, runtime requirements, assets and export
identities. The existing overnight launcher calls it before ordinary production
vision. It allows at most 90 minutes per night for at most 14 nights, progresses
through verification, resumable training, reload validation and paired validation
benchmarks, and then waits for coordinating quality review. Explicit candidate acceptance
freezes the reviewed configuration before admitting the sealed test comparison;
the test results require a second semantic review. Failed stages retain
receipts; gates and configuration changes are explicitly reported. No adapter is
promoted by exact-match proxies or a successful training process.

The remaining overnight time continues daily Q8 coverage. After semantic review,
the locked test comparison and three-night candidate trial remain separate gates.
The study's scheduled state does not establish that any training has run.

## Coordination checkpoint

- Source identity: canonical primary checkout; source-only public publication route.
- Review wave 1: training/data and Apple Silicon inference, GPT-6.1 Sol xhigh.
- Review wave 2: evaluation and chronicle architecture, after wave 1 is accepted.
- Goal-backed implementation wave 1: training harness and resident worker, each
  with separate source/test ownership. Wave 2: evaluation and chronicle integration.
- Root owns telemetry, reference review, evidence packs, exports and scheduling.
- Reviewed references: 100 viewed across six completed days; 60 train, 20
  validation, 19 eligible test. One test observation failed direct privacy review
  and is withheld. A replacement would require expanding the review permission.
- Added 60 controlled synthetic training examples; they never count as real-test
  accuracy. Exports separate the sealed test file from trainer inputs.
- Quantized MLX 9B assets: revision
  `938d8919941c6e7efd3c7150eff7fe9d12afa631`, approximately 5.98 GB,
  downloaded and SHA-256 verified. Production Q8 remains unchanged.
- Installed isolated MLX-VLM 0.7.6 environment; exact dependencies are pinned in
  `requirements-ml.txt`. No real model load or training has run in this stage.
- Four helpers remain available for goal-backed refinement in waves of two.
  Source preflights found and root fixed native MLX RNG restore, dropped pixel
  budgets in MLX-VLM's input wrapper, missing Torch/Torchvision preprocessing
  dependencies, loss-memory overhead, controller budget/state races, and exact
  chronology/correction boundaries.
- Metadata-only and synthetic CPU checks passed: native RNG replay, gathered
  assistant loss/gradient parity, and image budgets. All 120 train and 20
  validation examples were prepared locally without loading weights: maximum
  sequences 1,887/1,877, maximum 504 visual tokens under the 512-token cap.
- Per-attempt history covers production vision, resident startup/requests, text,
  and each training-validation generation. Retry identity includes request
  parameters and image/context; whole-device GPU remains separate from model
  process and MLX allocator measurements.
- Each night reserves at most 90 minutes durably before spawning. Crashes consume
  the unresolved reservation, and controller state is re-read under the lock.
  Ordinary daily coverage retains the rest of the vision window.
- Reference review permission is exhausted at 100 distinct images. Reuse their
  existing evidence for semantic assessment; no additional image upload is
  authorized. Twenty validation and nineteen sealed test examples are a pilot.
- Model training/promotion: not yet run; do not describe scheduled work as successful.
- Exact Oct 4 readback found 6,883 foreground episodes fragmented by polling
  jitter. Grouping matching context across at most two seconds reduced this to
  784, preserving exactly 34,452.544758 observed seconds. Gaps remain unallocated;
  anchors inside gaps and context/state/source changes cannot acquire support.

## Continuation and release checkpoint

Finite heartbeat `qwen-specialization-overnight-study` inspects every two hours,
quietly during daytime and with aggregate updates during active overnight work.
It reengages the four existing helpers with goal-backed assignments in pairs;
private examples stay restricted to the coordinating review scope. The overall
goal stays unfinished until the actual experiment and three-night recommendation.

The shared native launcher now installs an exec-surviving process timer. A
synthetic coordinator-crash test and macOS sandbox-exec probe confirmed termination
and lock release without relying on the coordinator. Failed worker receipt IDs
and raw-final safety flags remain linked in benchmark failure records.

The preflight checks establish software/API preparation. They do not establish
real 9B loading, Metal memory peaks, gradients, adapter quality, or accepted
production rollout. AC/resource/window gates still control admission.

Final preflight: `tools/check.py` passed 438 synthetic tests plus lint and source
syntax. A pre-inference time refusal remains pending rather than becoming a
permanent model error. Verified canonical aliases agree; conflicting canonical
identities abstain. The untouched prepared configuration was explicitly refrozen
after these reviews, preserving its earlier configuration and recording zero
started nights. Actual training, gradients on the 9B model, semantic comparisons
and the three-night trial remain pending.
