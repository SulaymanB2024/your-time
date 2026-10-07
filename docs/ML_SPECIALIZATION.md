# Qwen3.5 specialization

## Acceptance contract

Improve accurate project/task identification and daily temporal coverage using the
existing Qwen3.5-9B baseline. Timings alone do not establish task quality. Recorded
durations remain collector measurements; model outputs remain evidence-linked
inferences. Activity, screenshots, annotations, adapters and telemetry stay in the
private local state directory and never enter Git or hosted CI.

Private reference review requires explicit authorization. Reviewer access is scoped;
source reviewers receive code and aggregate measurements only. Keep other apps open. Heavy experiments run 00:30–07:00 America/Chicago; the
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
benchmarks, and then waits for coordinating quality review. A separately frozen
[runtime exploration](ML_TUNING.md) compares single-factor settings in two
counterbalanced rounds, with independent grading of each output. The sole
nominated setting receives a new full validation comparison before acceptance;
an individual quality or safety regression retains the original controls.
Explicit candidate acceptance
freezes the reviewed configuration before admitting the sealed test comparison;
the test results require a second semantic review. Failed stages retain
receipts; gates and configuration changes are explicitly reported. No adapter is
promoted by exact-match proxies or a successful training process.

The remaining overnight time continues daily Q8 coverage. After semantic review,
the locked test comparison and three-night candidate trial remain separate gates.
The study's scheduled state does not establish that any training has run.

## Runtime evidence

Execution checkpoints, training progress, reviews and acceptance receipts belong
in private local state. No user-derived runtime measurements or private review
status are published here. A loaded schedule, passing synthetic tests or finished
training process does not establish semantic quality or deployment acceptance.
