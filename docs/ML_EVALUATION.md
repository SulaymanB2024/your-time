# Private paired specialization evaluation

`specialization_benchmark.py` implements the evaluation component of the finite
study. Import and status handling load no model and read no reference examples.
Only the coordinating reviewer may inspect private examples and blind outputs.
Specialists receive source, synthetic tests and aggregate receipts only.

## Interfaces and private files

- `run_validation(config: dict, study_root: Path, budget: int) -> dict` is the
  study-orchestrator entry point. Budget is seconds, limited to 5400. Its public
  receipt has `status: complete|partial`, counts, proxy scores, group intervals,
  timings and public failure codes. `complete` means all selected variant/example
  pairs have terminal outcomes, including failures; it never means quality passed.
- `freeze_candidate(config, study_root, *, candidate,
  reviewed_assessment_sha256)` is an explicit coordinator operation after a
  complete blind validation assessment. It reads no sealed test. It refuses an
  incomplete review, changed experiment, unavailable adapter, and alteration of
  an existing freeze. Freezing authorizes a fixed evaluation candidate, not rollout.
- `run_locked_test(config, study_root, budget, *, frozen_candidate_sha256)` checks
  the immutable freeze, exact final experiment, validation results and reviewed
  assessment before permitting test access. It resumes the same frozen four-way
  comparison. A changed prompt/checkpoint/configuration cannot reuse the test run.
  This API must not be called for prompt, threshold or checkpoint selection.

Input exports are `DATA_ROOT/export/validation.jsonl` and, only through the locked
API, `DATA_ROOT/sealed/test.jsonl`. Rows require real source class, a unique ID,
one image path and SHA-256, context, episode ID, split, and user/assistant messages.
The assistant message is the coordinator-reviewed reference JSON. The harness
never reads the private reference manifest or train export. Export count and
content must match `config['exports'][split]` (`count`, `sha256`). The present
corpus has 20 validation and 19 eligible sealed test examples; one reviewed test
example was withheld for privacy. The 60 real plus 60 controlled synthetic training
examples do not contribute to held-out accuracy.

Private outputs under `study_root/benchmark/{validation,test}` are:

- `results.json`: frozen specification and every terminal attempt, including
  failures, blocked raw answers, telemetry pointers and interrupted work.
- `blind-assessment.json`: randomized opaque output keys, private image/context
  pointers, outputs and empty grades. It omits variant names and reference answers.
  A separate `blind-key.json` holds the variant mapping; keep it closed while grading.
- `summary.json`: aggregate-only receipt; paths, IDs, activity and model text are
  excluded. Missing raw safety evidence is null, not zero.

Use private atomic writes (0600 files, 0700 leaf directories), serialize through a
private benchmark lock, and retain the model APIs' shared inherited inference lock
and network sandbox. No package install, model download, publication or server is
part of this module. Heavy execution is admitted only during 00:30–07:00
America/Chicago and within the supplied finite budget. Keep other applications
open. Resource gates and per-call deadlines remain active; synthesis time is reserved.

## Controlled comparisons

| Variant | Settings and comparison |
| --- | --- |
| `production_q8` | Existing image-only caption prompt and Q8 runner. Derive first-attempt generation tokens/timeout from `vision_fallback.generation_budget(0)`. Default side 1600, image tokens 1024, threads 4, enabled thinking template and runner-default seed. No automatic retries. |
| `context_q8` | Identical Q8 settings, weights and projector; replace only the prompt with the frozen contextual prompt and request structured parsing. |
| `mlx_base` | Unchanged quantized MLX 9B with the contextual prompt; defaults side 1600, image tokens 512, context 4096, max generation 768, thinking false and fixed study seed. |
| `mlx_adapter` | Exact MLX base settings with the validation-proxy preselected adapter. If the proxy selects the unchanged base, independently compare the final completed epoch's trained adapter, explicitly recording `final_epoch_experiment_requires_semantic_review`. Proxy selection never establishes semantic quality or deployment acceptance. |

The original Q8 pair isolates prompt/context; the MLX pair isolates training. Q8 versus MLX
is a combined quantization, runtime, thinking and preprocessing comparison. Its
quality difference cannot be attributed to LoRA. The existing Q8 runner has no
explicit seed/thinking/context-length controls; those actual limitations are
included in the frozen spec. Image/resolution/thread and MLX thinking ablations
require separately fingerprinted paired studies. Q8 options have explicit `q8_`
prefixes, so MLX budgets cannot silently weaken production.

The [runtime exploration](ML_TUNING.md) adds separate, validation-only studies
after original baseline review. Its nominated settings must pass a new full
validation comparison and semantic review. In that final comparison, a changed
`context_q8` thread setting combines context and thread changes relative to
`production_q8`; use the preserved original comparison to isolate prompting.
Production Q8 controls remain identical. Candidate freezing binds the accepted
runtime decision and its exact original or confirmation review location.

Resume identity covers the full study configuration, actual source hashes,
production/context prompt template, per-example prompt/context/reference/image,
model/projector manifests, engine identity, pinned MLX versions and Python,
adapter snapshot, and preprocessing/generation settings. Actual Q8 weights are
reverified with bounded private reads; the worker verifies its complete MLX asset
closure. Changed identity is a stop, not cache reuse. Terminal failures remain
in the selected denominator and are not automatically retried. A crashed running
entry becomes interrupted on the next locked resume.

Execution balances small variant blocks and closes the resident worker before
another backend starts. Request timings separate from startup receipts. Capacity
uses total session wall time, including startup, hashing, failed requests,
instrumentation and teardown; summing warm successful request time is insufficient.

## Scoring and coordinator assessment

Automatic normalized exact project/task matching is a **literal proxy**. It does
not establish semantic equivalence, evidence entailment, real-project identity,
intent or accomplishments. Null project and null task are independently valid
references. The caption baseline has no invented structured labels and requires
human scoring for its paired quality comparison. Citation structure is validated
by the shared result parser; citations alone cannot pass semantic support.

Blind grades are booleans for independent project and task correctness, overall
evidence support, appropriate uncertainty, privacy disclosure, and unsupported
completion. `claim_support` additionally records support for each substantive
field (`fully_supported`, `partial`, `unsupported`, `not_asserted`); notes should
explain ambiguity or disagreements. Failure cannot receive a correct grade.
The coordinating reviewer assesses visible evidence, including stale metadata
and task switches, without treating unseen intent as truth.

Both inference APIs may supply `raw_safety_checks` with `available`,
`sensitive_keyword`, `email`, `url`, and `unsupported_completion`. These are
boolean-only checks of the final raw answer before cleanup, not retained raw
reasoning. Any true violation becomes a blocked failure in the denominator.
Exceptions may carry these checks too. The harness also supports optional raw
text in synthetic adapters for screening tests but does not save it. Keyword
screening has false positives/negatives and does not classify personal names or
all sensitive content; retain manual privacy and completion grades. Unavailable
raw checks cannot establish safety, and filtered safe images cannot establish
performance on sensitive screens.

| Dimension | Proposed acceptance gate for a larger representative evaluation |
| --- | --- |
| Project/task | Released project precision ≥95%, task precision ≥90%, answerable joint recall ≥80%. Include failures and correct nulls in separate all-selected accuracy. |
| Evidence | All citations valid and ≥98% emitted substantive claims fully supported. Grade claim meaning, not word overlap. |
| Abstention | ≥95% correct abstention on unanswerable fields; separately report missed answerable fields and failures. Compare quality at equal answer coverage. |
| Calibration | Report empirical correctness for supported/partial/unclear. These are ordinal categories, not probabilities. Numeric calibration is unmeasured until a separately specified confidence output and adequate holdout support Brier/reliability analysis. |
| Continuity | Labeled sequences: false switches ≤5%, true transition recall ≥90%, transition delay ≤one capture interval. Isolated selected frames cannot establish continuity. |
| Coverage/time | ≥90% eligible occupied 30-minute bins represented with supported labels; no credited unseen gaps/double-counted seconds; wrongly attributed seconds ≤5% on labeled intervals. Selected-example bin counts are not day-wide coverage or duration estimates. |
| Account rollups | Day/month/year sums reconcile with collector seconds; preserve aliases, missing-day coverage, independent phone overlap and confirmed-outcome provenance. This harness does not run the rollup system. |
| Privacy/completion | Zero observed sensitive disclosure or unsupported accomplishment claims; report blocking and raw availability separately. |
| Runtime | ≥99% valid results and required nightly episode coverage with a 20% reserve for three nights, all applications open; inspect responsiveness, thermal limits and swap growth. |

Do not collapse these independent gates into a single weighted score that lets
speed compensate for private disclosure or invented accomplishments. Require
coordinator semantic acceptance before any rollout; no API here promotes a model.

## Uncertainty and pilot limits

The summary reports paired wins/losses/ties and an episode-cluster percentile
bootstrap of the paired difference. Resample whole supplied episode groups,
keeping paired outcomes together; no interval is claimed with one group. Frame
counts do not multiply independent evidence. Episode IDs remain supplied metadata
and need a genuine episode/leakage audit. Duplicate image groups are counted
within each loaded split; the harness deliberately does not open training or
sealed test to claim a cross-partition audit.

Wilson intervals accompany all-selected literal proxies. Partial runs have
pending observations and provisional comparisons. Human paired scores use only
fully assessed pairs and report their actual pair/group count; they cannot hide
unreviewed cases. For longer trials use whole-day resampling with episodes nested
inside days. The current single-day validation/test partitions cannot estimate
variation between days, general accuracy, calibration, or month/year usefulness.

Zero failures among 20 independent cases leaves a one-sided exact 95% upper
failure bound of 13.9%; for 19 it is 14.6%. Approximately 299 independent
zero-failure observations are needed to bound failure below 1%. Correlated frames
provide weaker evidence. References: [NIST confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[selective prediction](https://proceedings.mlr.press/v97/geifman19a.html), and
[confidence calibration](https://proceedings.mlr.press/v70/guo17a.html).

For the pilot, a ≥5-point paired semantic joint gain with no critical regression
justifies further evaluation. Production adapter promotion needs that gain over
the unchanged MLX base, a positive lower bound on the paired improvement in a
larger representative holdout, all gates, and the three-night trial. Opening the
sealed test forbids adapting the candidate from its failures. Further real-example
review beyond the authorized corpus requires additional authorization.

## Verification and required scenarios

Focused synthetic checks cover stale context, independently nullable fields,
raw flags before cleanup, unavailable raw measurements, invalid schemas, failed
denominators, cluster dependence, unique temporal bins, daylight execution refusal,
exact resume identity, changed context, preserved failures, blind scaffold privacy,
review-required freezing, and refusal to open test for changed candidates.

Additional coordinator sequence/rollup checks must cover same-title task switches,
cross-app continuation, near-identical layouts with changed project text,
conflicting propagation, midnight and DST, idle/lock interruptions, missing frames,
multiple displays, concurrent phone use, semantic aliases, and identical task names
in different projects. No new private sequence set is acquired by this component.

Implementation verification on 2026-10-05: 64 synthetic tests passed across
`test_specialization_benchmark.py`, `test_specialization_study.py`,
`test_vision_quality_eval.py`, and `test_specialization_worker.py`; focused Ruff
checks passed. These checks exercised fixtures and fake inference backends only.
They did not open the private corpus, run a model, measure hardware capacity,
establish semantic accuracy, freeze a real candidate, or open the real sealed test.
