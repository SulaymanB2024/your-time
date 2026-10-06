# Paired runtime exploration

`specialization_tuning.py` explores throughput settings after the coordinating
reviewer grades the original four-way validation comparison. This is an
experiment, not an automatic deployment or a demonstrated speed improvement.
Private outputs, grades, timings and choices remain outside Git.

## Controls

The reviewer freezes either the unchanged MLX base or the trained adapter as the
provisional runtime backend. Twelve predeclared conditions change one factor:

| Backend | Conditions |
| --- | --- |
| MLX | Reference; worker lifetimes of 1 or 8 instead of 4 requests; thinking enabled with a 256-token budget; image sides 1,024 or 2,048 instead of 1,600; visual-token limits 1,024 or 1,536 instead of 512; central 80% crop. |
| Q8 context | CPU threads 2 or 8 instead of 4. Other Q8 settings remain fixed. |

Image side is a preprocessing request, not proof of effective model resolution.
The worker records actual processor grid dimensions and visual-token count when
available. The crop is a controlled experiment with a known field-of-view loss:
it removes 10% from each edge in memory, leaves the original image unchanged,
and marks that image scope in the evidence pack. OCR and metadata still cover the
full observation. It is not a detector of the user's focus or a readable-text ROI.
Original and cropped outputs receive independent semantic review.

All MLX comparisons retain the same frozen weights, adapter, generation limits
and seed. Thinking uses the supported chat-template switch. Q8 exposes threads,
resolution and token controls; its existing template/seed limits remain explicit.
Do not assume an MLX result establishes llama.cpp performance.

## Grouping and measurement

Use the same validation examples in every condition. A transitive grouping of
episode IDs, identical images and near-image groups reserves up to eight examples
for exploration and the remainder for confirmation. Groups never split to obtain
an attractive sample count. If no useful partition exists, all validation is
reused, and the receipt explicitly states that confirmation is not independent.
Validation already influenced checkpoint quality; neither partition is a new
generalization holdout. Sealed test examples remain unopened.

Run two rounds, reversing condition order and example order in the second.
Repeated generations do not increase the number of independent semantic
examples. Each condition/round has its own immutable specification, results,
blinded grades and wall-time journal. Changes in source, settings, examples or
review invalidate reuse.

Full cell wall time includes startup, image checks, inference, failures,
instrumentation and teardown. Admission reserves the whole intended worker block,
including per-request control and sampler margins. Partial or failed lifetimes
cannot serve as clean runtime evidence, even if remaining frames later finish.
Unresolved crash costs remain unknown with a recorded reservation bound.
Completed cells are verified and skipped on resume; resume bookkeeping stays in
the outer sweep journal rather than changing their comparison costs.

## Selection and confirmation

Every output needs coordinating semantic grades and claim support. A setting
can be nominated only if it gives at least 15% full-wall improvement in **both**
rounds, all frames finish, and no reviewed example loses project correctness,
task correctness, evidence support or appropriate uncertainty. Zero unsupported
claims or observed critical safety flags are required. Aggregate improvements
cannot cancel individual regressions. If no condition qualifies, keep the
reference. Choose at most one MLX factor and one Q8 thread setting.

Freeze that sole nomination before a new four-way comparison on all validation
examples. Review the reserved confirmation examples separately and check quality
and safety across the entire validation set. A regression retains the original
controls; do not choose another setting using confirmation results. Both rounds'
speed evidence is provisional: later sealed evaluation and three-night full-wall
coverage determine the deployment recommendation and sustainable capacity.

The original comparison remains unchanged. Final production Q8 also keeps its
original controls; a tuned context-Q8 comparison may combine context and thread
changes, which must not be presented as an isolated prompt effect.

The accepted runtime decision binds the complete original baseline, frozen plan,
all 24 result/review pairs, exact nominated controls and confirmation hashes.
Candidate freezing, sealed evaluation and shadow trial recheck that identity.
No tuning path reads sealed examples or modifies production captions.

## Coordinator stages

1. Grade original validation; `begin-tuning` freezes a provisional MLX backend.
2. `run` performs bounded sweeps; `accept-tuning` requires every reviewed cell hash.
3. `run` performs full confirmation; `accept-confirmation` freezes its reviewed decision.
4. Grade the accepted original or confirmation comparison; `accept-validation`
   freezes the final model/configuration before sealed test access.
5. Review sealed results, then execute and review the separate three-night trial.

The study uses the existing finite overnight budget, locks, network denial,
resource gates and 07:00 inference cutoff. Import/status operations load no model.
Only the coordinating chat may inspect private outputs; specialist reviews use
source, synthetic tests and aggregate receipts.
