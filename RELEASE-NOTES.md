# Release Notes

## 0.6.1 — 2026-06-12

- **Budget persona crash fix.** The Budget view built its default GPU config
  with `buildGpuPresets(...)[0]` and dereferenced it unconditionally
  (`gpuConfig.vramPerGpu` / `.gpuCount`). In regions whose catalog has no pod
  large enough for the current top models — e.g. North America / Asia Pacific /
  Global, where the only {1,4,8}-count pods are 80 GB-class — `buildGpuPresets`
  correctly returns `[]`, so the default config was `undefined` and the view
  threw `TypeError: Cannot read properties of undefined (reading 'vramPerGpu')`.
  `BudgetFlow` now treats the config as nullable, short-circuits the chart memo,
  and renders "No GPU configuration in {region} can host the current top
  models" instead of crashing.
- **Region switch no longer keeps a stale GPU config.** Switching region now
  re-derives the default GPU config — preserving the current selection when it
  is still offered, otherwise falling back to the new region's default (or the
  empty state when none can host the top models).
- **Regression test.** Added `gpu-presets.test.ts` covering the empty-region
  case (`buildGpuPresets` returns `[]`) that produced the crash.

## 0.6.0 — 2026-06-03

- **Clearer trend-chart copy.** Retitled the cost-trend chart to "Is It
  Getting Cheaper to Self-Host the Best Open-Source Model?" and the
  efficiency chart to "Are API costs per task lowering?" (tab shorthand
  "Efficiency" → "API costs"), with a subtitle clarifying these are
  API-metered (pay-per-token) costs rather than self-hosting. The scaling
  chart subtitle now reads "Monthly self-hosting costs…" to disambiguate it
  from the API-metered view.

## 0.5.1 — 2026-05-11

- **Snapshot dedup baseline fix.** `run_snapshot_export` now walks existing
  + new dates in chronological order and seeds each new date's dedup/diff
  baseline from the snapshot that *chronologically* precedes it, instead of
  always from `max(existing_dates)`. The old behavior wrote bogus duplicate
  snapshots (and produced misleading "new models" / score-change entries in
  the notification email) whenever a new date landed between two existing
  ones — which is exactly what happened after the 0.5.0 timezone-fix regen
  dropped some local-time dates from the index.
- **Orphaned snapshot cleanup.** Removed `2026-05-02`, `2026-05-04`, and
  `2026-05-08` from `web/public/data/snapshots/` — all byte-identical
  duplicates of their chronological predecessors, left over from the
  earlier buggy run.

## 0.5.0 — 2026-05-10

- **Snapshot pipeline timezone fix.** `git_repo.py` now runs all git
  subprocesses with `TZ=UTC` and uses `--date=short-local`, so commits are
  bucketed by the same UTC date that the `--until` filter uses. Previously a
  cross-midnight upstream commit (e.g. GPT-5.5's `commit0` result) could be
  classified into a date but excluded from that date's snapshot, leaving the
  model permanently below the per-lab "best" picker.
- **API picker now prefix-based.** `MODEL_LAB_MAP` allowlist replaced with
  `LAB_PATTERNS` prefix matching, so new models from Anthropic / OpenAI /
  Google are picked up automatically without code changes.
- **Snapshot dedup.** The exporter now skips writing a new snapshot when its
  content is byte-identical to the previous one — upstream commit-days that
  only carry metadata churn no longer mint duplicate snapshots.
- **Email guardrails for tests.** A pytest autouse fixture
  (`block_real_smtp`) blanks SMTP env vars and replaces `smtplib.SMTP` with a
  raising stub, so tests can never accidentally send real notification email.
- **Snapshot matrix fixes.** Coverage matrix corrections for edge cases in
  per-model category visualization.
- **New model coverage.** Kimi-K2.6 added to the leaderboard.

## 0.4.0 — 2026-04-27

- **API vs Self-Hosting chart redesign.** Step-function self-hosting curve,
  capacity-aware props, moved into the Budget persona view. End-of-line
  labels for closed model curves; tooltip entries sorted most→least
  expensive; fixed-axes so scale doesn't shift when pickers change.
- **Capacity chart.** New zoom in/out controls, tweaks to controls and
  legends, capacity-aware self-hosting helpers in `api-hosting-cost.ts`.
  "Turn" terminology renamed to "request" throughout.
- **Snapshot Matrix Dialog.** New UI component for visualizing per-model
  category coverage of the OpenHands Index, with model filtering logic and
  tests. Wired into the header alongside reordered action buttons.
- **Per-model snapshot coverage emails.** `notify_new_snapshots` reports
  which models gained categories on each upstream update, surfaced through
  `main.py` to the existing email pipeline.
- **Dependabot housekeeping.** Bumps for `actions/checkout`,
  `actions/setup-node`, `actions/upload-pages-artifact`,
  `actions/deploy-pages`, `astral-sh/setup-uv`, and `setuptools`.
- **Model-name fixes** for entries that diverged from upstream canonical
  names.

## 0.3.0 — 2026-04-22

- **LiteLLM API pricing pipeline.** New `litellm_source.py` fetches official
  per-token pricing for the best closed model per lab, exported to
  `api_pricing.json` for the frontend.
- **API pricing frontend integration.** New `ApiPricingEntry` type,
  `useApiPricing` hook, dev fixture, and `api-hosting-cost.ts` utility
  library powering cost calculations.
- **API vs Self-Hosting chart (initial version).** First implementation
  with model picker, percentage-of-performance overlay, vertical lines, end
  labels, and a wider cache hit rate selector. Multiple iterative
  improvements to axis ranges, curve response to cache hit rate, and tooltip
  ordering. Lab display names normalized via `labToDisplayName`.

## 0.2.0 — 2026-04-17

- **Snapshots redesign.** Reworked snapshot generation approach for the
  OpenHands Index integration; more reliable update-date handling globally
  surfaced from the pipeline.
- **GPU specs source switched to dbgpu.** Replaced previous mechanism;
  removed published-params and unused legacy code paths. Reviewed throughput
  formula for MoE architectures; added cap on max tensor parallel for
  non-NVLink GPUs. Projected GPU setup calculations added.
- **Budget persona.** Redesigned around team capacity; refined chart
  controls and copy.
- **Operational hardening.** New breaking-format-change alert; new alert for
  unsupported HF architectures; Nemotron architecture support; new HF
  mappings; CI for tests; subdomain hosting fix; emails suppressed for GPU
  updates.
- **Frontend polish.** Better small-screen layout; tok/sec overlay; GitHub
  repo link; updated screenshots.
- **Build & ops.** `cloudbuild.yaml` added.

## 0.1.0 — 2026-02-18

- **Initial commit.** Project scaffold: Python pipeline, Next.js web app,
  and the OpenHands Index results submodule.
