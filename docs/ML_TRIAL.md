# Three-night shadow trial

The coordinating review freezes one candidate using the validation comparison.
The sealed test measures that choice; it cannot be used to choose another model.
After a complete held-out review, `specialization_study.py accept-test` freezes
the test results, review, candidate, source and runtime identities. A changed
identity requires a new experiment rather than silently resuming.

Before the first run only, a reviewed source amendment can use the explicit
`refreeze-unstarted` action. It archives the prior configuration and refuses any
recorded night, experimental artifact, or changed asset/dataset. Executed studies
cannot use this action to reuse results with new code.

The trial runs locally in the study's existing 90-minute nightly budget, inside
00:30–07:00 America/Chicago. It uses the exact benchmark backend and decoding
configuration. Each cohort covers a completed local day, with selection spread
across occupied half-hour bins and activity episodes. OCR availability, sensitive
filters, missing images, metadata conflicts and collector support are reported as
separate exclusions. Whole-day denominators are frozen before frame selection.

Private journals reserve runtime before preparing a cohort. Crashes retain an
unknown measured duration and a conservative reservation bound. Resume checks the
cohort and each frame's identity; completion is recalculated from those records,
not cached summaries. Pre-inference resource refusals remain pending rather than
being counted as failed model attempts. Actual failed attempts retain telemetry
links and boolean safety evidence without storing diagnostic bodies.

Three distinct nonempty cohorts and three nights with successful generations
are required. Resuming one cohort on three nights does not satisfy that gate.
Empty or failed cohorts remain in runtime accounting. Runtime quota estimates
include startup, preparation, cleanup and failures, with a 20% reserve.

Coverage measures valid frames, occupied bins, unique episodes and the union of
observed support intervals. It does not turn gaps into recorded time. Structural
validity and a model's supported claim do not establish semantic accuracy on new
frames: those measurements remain explicitly unavailable without new reference
labels. Pilot test accuracy is shown separately and carries sample uncertainty.

`completed` counts accepted generations, not completed work. A completed bin
contains at least one such observation; it does not establish thirty minutes of
attention. `model_claimed_supported_seconds` also includes activity-type claims
with unknown project/task candidates, so it is not accurate project/task coverage.

The recommendation is a private review artifact. The trial never changes
production captions, task allocations or the dashboard, and does not promote an
adapter automatically. Root reviews quality, coverage, failures, resource use and
full runtime before giving a deployment recommendation.
