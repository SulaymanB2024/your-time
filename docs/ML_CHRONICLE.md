# Chronicle episode integration

`activity_episode_store.py` is a standalone component for root integration. It
does not discover projects, query the capture ledger, run models, promote a
model, or establish completed outcomes. Only synthetic fixtures were used to
implement and verify it. Root now supplies `chronicle_activity.py` and integrates its conservative
summary into `chronicle_rollup.py`; the component itself never promotes models.

## Acceptance contract

Store fingerprint-bound `activity_result_v1` hypotheses; split episodes at
context/title/state/source changes, idle/lock, gaps, explicit context boundaries,
and support barriers; allocate each observed foreground second at most once.
Never attach background file writes by timing, infer attention, or promote an
inference to a confirmed accomplishment. Model allocation defaults off and
requires explicit semantic review plus a verified three-night trial receipt.

The three owned surfaces are this document, `activity_episode_store.py`, and
`tests/test_activity_episode_store.py`. No model runs, private-data reads,
installs, publishing, commits, or overlapping-source edits are part of this
component's verification.

## Exact caller interface

1. `build_episodes(samples, *, context_boundaries=(), max_seconds=900, max_gap_seconds=0)` accepts
   disjoint observed runs with `id`, aware `start_utc` / `end_utc`, `state`
   (`active`, `unattributed`, `idle`, `locked`), and exact `sampled_seconds`.
   Optional `app`, `window`, `context_key`, `evidence` / `source`, and
   `support_barrier` establish boundaries. By default every gap breaks continuity.
   The collector bridge uses `max_gap_seconds=2` to group matching context across
   polling jitter, retaining every exact disjoint observed run. Missing seconds
   remain unallocated, and observations inside a gap cannot anchor an inference. Explicit
   aware timestamps in `context_boundaries` split reused titles without carrying
   a label. UTC normalization handles repeated local DST hours.

   **Sparse segments must supply `support_runs`:** a list of exact observed
   `start_utc` / `end_utc` subintervals whose durations sum to `sampled_seconds`.
   Existing `daily_analysis.mac_segments` can merge polling gaps; elapsed span
   is therefore not proof of continuous support. Recover trimmed atomic runs
   from the deterministic source layer before calling this API. No proportional
   smearing into missing time is accepted. Input overlap is rejected; resolve it
   in that source layer before construction. Episode IDs bind complete sample
   topology, so backfilled or changed context invalidates affected inferences.

2. `make_fingerprints(context, *, screenshot_sha256, input_sha256,
   engine_sha256, prompt_sha256, model_sha256, adapter_sha256=None)` returns seven
   exact fingerprint fields. All non-null fields must be full SHA-256 digests.
   `context_sha256` hashes the **full** canonical context, including its existing
   `input_sha256`; it is not interchangeable with `activity_context.pack`'s
   differently serialized input hash. `input_sha256` binds the exact request and
   inference/integration configuration, including the current identity binding
   registry revision; the caller must derive it independently. Engine
   identity must bind executable/dependencies, prompt identity must bind actual
   prompt/template, and model/adapter identities must bind verified snapshots.

3. `make_inference(*, episode, observation_id, timestamp_utc, context, result,
   fingerprints, project_bindings=(), task_bindings=(), radius_seconds=30)`
   validates the current activity schema and four-field `claim_evidence` mapping.
   It requires exactly one primary observation with matching ID and timestamp,
   verifies the context hash, and bounds support to at most 30 seconds each side
   of that observation, clipped to one foreground episode. A support barrier
   prevents inference attachment. This produces an immutable
   `activity_episode_inference_v1` record, always `model_hypothesis`, with empty
   `confirmed_outcomes`. `persist_inference(store_dir, **same_arguments)` writes
   it atomically as a private per-record JSON file. The directory is explicit;
   there is no default route to live private state. Raw OCR/context is not
   duplicated: source evidence is retained as IDs/types/digests, alongside the
   model result and its claim citations. Citations remain references, not
   semantic entailment proof.

4. `read_fresh_inferences(store_dir, *, episodes, current_evidence,
   runtime_fingerprints)` returns only matching records. `current_evidence` is
   keyed by observation ID, with current aware `timestamp_utc` and three fields:
   `screenshot_sha256`, `input_sha256`, `context_sha256`. The runtime dictionary
   contains `engine_sha256`, `prompt_sha256`, `model_sha256`, `adapter_sha256`.
   **Derive these from current verified sources, never from stored records.**
   Missing evidence, changed OCR/context/image, changed runtime, changed episode,
   modified records, widened support, and asserted confirmed outcomes fail
   closed. Missing stores are read without creation. Private owned regular
   files, single links, canonical directories, and record size bounds are checked.

5. `allocate(episodes, inferences=(), corrections=(), *, current_evidence,
   runtime_fingerprints, reviewed_record_ids=(), promotion=None)` rechecks all
   fingerprints and source support, then emits exact intervals and totals.
   Corrections have `id`, aware `start_utc` / `end_utc`, `label`,
   `evidence_tier='user_confirmed_label'`, and `revision_sha256` computed by
   `correction_fingerprint(current_correction)`. Optional `sample_ids` narrows
   their exact support; optional canonical `project_id` / `task_id` survive
   allocation. The caller fetches **current** correction revisions and removes
   retracted records. Hashing cached corrections cannot establish freshness.

   Precedence is current user correction **only on unattributed samples**, then
   an eligible model inference, then unknown. Conflicting user labels abstain
   and do not fall back to a model. Overlapping model supports use the nearest
   observation; equal-distance conflicting hypotheses abstain. Idle, locked,
   barriers, and missing gaps receive no task time. `partial` / `unclear` model
   results stay stored but cannot allocate specific task time.

   Model eligibility requires the record ID in `reviewed_record_ids` and an
   explicit promotion dictionary: `enabled=True`, nonempty `semantic_review_id`,
   at least three distinct ISO local dates in `trial_nights`, and full
   `trial_receipt_sha256`. **Root must independently verify the semantic review,
   trial results and promotion authorization before passing this dictionary.**
   This component checks receipt shape; it cannot prove external trial success.
   Review IDs bind immutable inference IDs, so changed results need fresh review.
   The default is disabled, even for structurally supported model results.

   Output `totals` reconciles `observed_sampled_seconds = foreground_seconds +
   nonforeground_seconds` and `foreground_seconds = allocated_foreground_seconds
   + unknown_foreground_seconds`. All arithmetic uses exact microsecond duration
   and Decimal accumulation; no device totals are combined. Intervals retain
   episode/sample IDs, evidence tier, selected inference/correction ID, uncertainty
   and canonical project/task IDs where supported. User-confirmed labels are
   labels, never confirmed results; model labels remain `supported_inference`.

## Supported identity binding

Project bindings are caller-verified dictionaries with `id`, `aliases`, and
`evidence_ids`. Use the canonical Git common-store identity or supported native
document/project identifier. A matching alias must be anchored by the **current
observation ID** in both the binding and project claim. File-only evidence is
insufficient, including a similarly named folder or a simultaneous background
write. Ambiguous alias matches produce no canonical identity.

Task bindings add `project_id` and require the resolved project plus a current
observation anchor. Task IDs are not project IDs or hashes of display labels.
Aliases survive in the persisted identity. No automatic label-based merging or
cross-app/day task continuity is introduced; root owns verified registry updates.

## Root integration still required

- Add a deterministic observed-run extractor rather than passing sparse merged
  focus blocks. Supply explicit context boundaries for observed task switches.
- Run the authorized worker with the exact context/request/runtime identities;
  persist successful results via this API. Keep stale/inferred metadata marked
  as hints in retrieval. Do not use a whole-day title as the inference scope.
- Build current evidence fingerprints from current sources and verified worker
  identity. Supply canonical project/task bindings only where current visible
  evidence supports them. Load current user correction scope/revision separately.
- In `chronicle_rollup.day_summary`, consume a single `allocate` result for task
  totals. Keep existing window/screen arrays as alternative evidence views, not
  additive seconds. Keep iPhone and calendar duration independent.
- Retain receipt/artifact IDs and link them explicitly to canonical tasks. Join
  `task_outcomes.active_outcomes` as separate confirmed evidence; no screenshot,
  file change or local commit alone establishes delivery.
- Fingerprint daily dependencies (source cutoff, episodes, evidence/runtime,
  correction revisions and outcomes). Refresh every affected day/week/month/year
  after changes, including days older than eight days. Preserve unknown time,
  classified-time coverage and narrative subset coverage in each period.
- Perform explicit semantic review and the three-night trial before production
  model attribution. Passing synthetic tests does not establish model accuracy,
  task usefulness, M3/24GB throughput, live integration, or successful promotion.

## Metadata-only status

`python activity_episode_store.py status --store-dir /absolute/private/store`
counts files/bytes and invalid metadata entries without deserializing source or
result content. It does not claim freshness or promotion. No live status command
was executed for this component.

## Verification checkpoint

Focused synthetic tests cover reused-title changes, corrections confined to
unattributed support, background project changes, stale OCR/image/context/runtime,
overlapping user/model allocation, aliases and identity collisions, DST, exact
missing gaps, support barriers, fractional durations, persistence integrity and
the disabled promotion gate. No private source was queried. Verification on
2026-10-05: `python -m pytest -q tests/test_activity_episode_store.py` passed all
39 tests; `python -m ruff check activity_episode_store.py
tests/test_activity_episode_store.py` passed. The existing repository `.venv`
was used without installs. Root integration and the real three-night trial
remain separate acceptance gates.

## Root integration checkpoint

`daily_analysis.observed_mac_runs` supplies exact trimmed collector intervals
shared by merged segments and episode consumers; polling gaps remain absent time.
Both daily views share one source-row snapshot. Current corrections retain their
stored boundaries and revision rather than extending into later observed time.
Capture/collector agreement uses a shared raw-metadata fingerprint, so sanitized
or truncated display hints cannot create false mismatch or false agreement.

Candidate structured results can be persisted explicitly through `persist_result`;
they do not enter production duration allocation. Daily and month/year summaries
carry exact episode totals with the model gate closed. Historical refresh checks
source/review/analysis content fingerprints, so old corrections, retractions and
backfills refresh dependent periods. Existing inferred window/screen views remain
alternative evidence views; they are not added to episode task totals.

## Aggregate production reach audit

`chronicle_audit.production_frame_reach(receipt, model_sha256=..., prompt_version=...)`
reconstructs one completed run's captures on its declared source day from a single
read-only SQLite snapshot. Model, prompt, capture day and update window are explicit
filters. Current caption rows can be overwritten; this reconstruction does not
replace immutable run receipts or establish historical completeness.

The aggregate report counts distinct capture observations, rejects changed or
unavailable images and capture/collector conflicts, and unions exact observed
support within thirty seconds of each eligible anchor. Polling gaps, idle/locked
time and support barriers receive no credit. Image checks use bounded reads of
private archive files with descriptor-relative nonblocking opens; symlinks,
nonregular files, hard links and unapproved paths are rejected.

Collector episode reach, supported seconds and occupied half-hour bins measure
different things. Support intervals can touch a neighboring bin; anchor-point bin
reach is reported separately for comparison with the frozen shadow-trial metric.
Identical image bytes at different capture times remain distinct observations;
surplus records for one observation are counted separately. Reports expose only
aggregates and source fingerprints. Semantic accuracy, attention and accurately
described task episodes remain unmeasured by this audit.
