# Release Notes

## 0.13.0 — 2026-07-16

- **GPT-5.6 is now the OpenAI model in the API-vs-self-hosting chart.** It has
  LiteLLM pricing but no OpenHands Index results yet, so the score-derived pass
  could only reach GPT-5.5 and the chart stayed a release behind. It is pinned
  explicitly and shows as an **unranked** row — real per-token pricing, no
  scores — until the Index publishes results, at which point the pin comes out
  and the normal selection takes over.
- **One source of truth for each lab's closed model.** The pipeline decides
  which closed model represents Anthropic, OpenAI and Google, and publishes that
  decision in `api_pricing.json`. The API-vs-self-hosting chart and the Snapshot
  Coverage Matrix now both render it, so they cannot drift apart. The web app's
  hand-maintained mirror of the selection logic is gone — there is nothing left
  to keep in sync.
- **Pinning is how you show a lab's *latest* model.** A lab's newest release is
  not always its highest scorer, and the automatic pick always prefers the
  highest scorer. `CLOSED_MODEL_OVERRIDES` in
  `pipeline/pipeline/sources/litellm_source.py` is the one place to override it.
- **Docs corrected where they had drifted from the code.** `UPDATE-MODEL.md`
  documented a `MODEL_LAB_MAP` that no longer exists, `pipeline/README.md` told
  you to mirror the override into the web app, and the pipeline's alert emails
  pointed at a section title that had been renamed away.

## 0.12.0 — 2026-07-06

- **New "GPU Node Price" chart in the Trends persona.** Tracks the cheapest
  monthly rental cost of an **8× GPU node** (B300, B200, H200, H100, A100 80GB,
  RTXPRO6000) over time, as a whole-node bundle price in `$/month`.
- **Snapshot-based series.** The history is a change-log rather than one point
  per day: a snapshot is recorded only when a node's cheapest price moves, and
  the x-axis is labelled at those snapshot dates. This cut the stored series
  from 109 daily points to 21 change points and stops the daily data bot from
  committing an unchanged row every day.
- **Readable overlay.** The hover overlay labels each node as `8× <GPU>` and
  sorts nodes most-expensive first.
- **No provider names recorded.** The history is a price record only — supplier
  names are no longer stored in the data or shown anywhere in the UI.

## 0.11.0 — 2026-07-02

- **API-vs-Self-Hosting chart reworked around cost per request.** It now plots
  cost per 1k requests against monthly volume: API models are flat metered
  lines, and self-hosting is a capacity-aware curve that amortizes the box's
  fixed cost over its throughput — falling as the box fills, flooring at the
  full-utilization cost, and stepping up as boxes are added. The self-hosting
  box is sized at its **scale** (aggregate batched-decode) throughput, not a
  single stream. Each break-even dot is labelled with the **box utilization**
  needed to beat that API model (e.g. "40% used"), the chart names the box and
  its monthly price, and the hover shows each option's total monthly cost at
  that volume.
- **"Models served" split by whether capacity is modeled.** Only architectures
  we can size at scale are listed with their throughput; architectures whose
  capacity we can't compute are called out separately.
- **Manual override of the API model shown per lab** in the cost comparison.
- **GPU presets use datasheet VRAM.** Preset labels and config now take VRAM
  from the GPU datasheet rather than an offering's occasionally-noisy reported
  value (so "8× B200 180GB", not "179.06…GB"), and presets are deduped per
  config.

## 0.10.0 — 2026-07-01

- **Performance persona recast around Fit and Scale.** The four concurrency
  tiers (Single Agent … Agent Swarm) are replaced by two operating points, both
  sized at the fixed 90% memory utilization the calculator assumes: **Fit** (the
  cheapest GPU setup that fits the model and serves ≥1 stream) and **Scale** (the
  cheapest that sustains 100+ streams). Each cell is now a compact card — GPU
  layout, monthly price, a single stream count, and single-stream/aggregate tok/s
  when the architecture is modeled.
- **Decode model gains the KV-cache read (F-5).** Single-stream decode now
  charges the decode-time attention read of the resident KV cache from HBM on
  every token (the full context, sharded across the KV-head TP group), on top of
  the F-4 op-chain / EP-dispatch / expert-imbalance terms. Prefix caching speeds
  prefill, not this read, so it is charged in full — sizeable on bandwidth-bound
  layouts, bringing paired-NVLink/PCIe decode close to measured. Concurrent
  streams are sized at whole-prompt admission (each reserves KV for its full
  context; prefix reuse no longer inflates the count).
- **Budget requests/hour is prefill-aware on aggregate throughput.** Serving
  capacity now models a request as a time-share of the node between prefill
  (compute-bound, discounted by a 90% prefix-cache hit) and decode (at the
  aggregate batched throughput, not N × single-stream). The capacity chart lists
  architectures it can't model separately from models that simply don't fit, and
  the tooltip surfaces aggregate tok/s under decode.
- **Interconnect tier is data-driven and shown consistently.** Every GPU carries
  one of three datasheet tiers — **NVSwitch** (fully-connected mesh),
  **NVLink+PCIe** (paired NVLink, pipeline across pairs), or **PCIe** — and the
  topology is taken straight from the GPU, never overridden by the UI. The custom
  GPU picker shows each GPU's tier and every config displays its interconnect
  badge (a single GPU shows PCIe).

## 0.9.0 — 2026-06-23

- **KV cache dtype setting (default 2 bytes).** The calculator hardcoded KV
  precision to `auto`, which resolved to FP8 (1 B) on Hopper+/Ada/Blackwell. But
  vLLM's `kv_cache_dtype=auto` follows the model's bf16/fp16 *compute* dtype, not
  the GPU's FP8 capability — so real deployments store bf16 KV (2 B) regardless
  of weight quantization. This over-predicted concurrent streams ~2× on
  Hopper-class GPUs (e.g. MiniMax-M2.7 on 8×H100: 57 predicted vs ~28 measured).
  Advanced Settings now has a **KV cache dtype** selector: **"2 bytes (FP16/BF16)"**
  (new default, always 2 B) or **"1 byte (FP8, Hopper+)"** (opt-in 1 B on
  FP8-capable GPUs only). The default flip halves stream counts on
  Hopper+/Ada/Blackwell; A100 and older are unchanged. Decode tok/s is
  unaffected (KV dtype doesn't enter the bandwidth roofline).
- **GPU interconnect inference.** gpuhunt exposes no interconnect field, so
  offerings were emitted with `interconnect=null` and the calculator treated
  every multi-GPU setup as PCIe (TP≤4, higher all-reduce penalty), systematically
  under-reporting throughput for SXM/NVLink datacenter nodes. The pipeline now
  infers `nvlink`/`pcie` per offering from the instance name: explicit "PCIe"
  wins, then SXM/HGX/DGX/NVL markers, then a datacenter-class fallback
  (A100/H100/H200/B200/B300/V100/GB200/GH200 default to NVLink since their PCIe
  variants are always labelled), otherwise PCIe. Single-GPU offerings stay null,
  and the default never over-estimates throughput.
- **Kimi-K2.7-Code added (unranked).** Adds `moonshotai/Kimi-K2.7-Code` (1T-param
  MoE, 32B active, MLA, INT4). It enters as unranked — sized and fit to GPUs, with
  no OpenHands Index score yet.

## 0.8.0 — 2026-06-17

- **Unranked models.** Models whose architecture and sizing are known but that
  don't yet have an OpenHands Index benchmark score are now first-class: they're
  sized, fit to GPUs, and surfaced across the capacity, size, and coverage views
  rather than dropped for lacking a rank. The Performance persona gets a
  dedicated **Unranked Models** section (a toggle, reusing the Top Models matrix
  UI) that omits the SOTA bar since there's no score to compare against. Adds
  **GLM-5.2**, **GLM-5.2-FP8**, and **MiniMax-M3-MXFP8** as unranked entries.
- **MXFP8 sizing fix.** `resolveModelPrecision` had no case for `"MXFP8"`, so it
  hit the conservative fp16 fallback (2 bytes/param). MiniMax-M3-MXFP8 (~426B)
  was sized at ~853 GB → 13 H100s and reported as not fitting 8×H100,
  contradicting the vLLM recipe page. MXFP8 is an 8-bit format (~1.03 effective
  bytes/param); it now maps to the fp8 bucket and sizes at ~426 GB → 7 H100s,
  which fits. Since this is the single sizing chokepoint, the fix propagates to
  every view. Added a regression test.
- **Default avg input tokens 40k → 50k.**

## 0.7.0 — 2026-06-17

- **MiniMax-M3 support.** M3 ships as a vision-language model (`model_type`
  `minimax_m3_vl`) with a new sparse text backbone, so the pipeline had been
  skipping it as an unsupported architecture and the catalog fell back to
  tentative figures borrowed from MiniMax-M2.5. It now resolves to its real
  architecture: **~427B total / ~23B active params, 1M context, BF16**. The
  pipeline unwraps the VL wrapper to the text backbone (which carries no
  `model_type` of its own), counts params with a dedicated counter (dense/MoE
  split from `moe_layer_freq`, distinct `dense_intermediate_size`, GQA with
  per-head QK-norm, MTP modules), and falls back to the top-level config for
  precision detection (the wrapper declares `torch_dtype` there).
- **MiniMax Sparse Attention (MSA) KV sizing.** New `attention_type` `"MSA"`.
  Per the M3 technical report, MSA's 1M context comes from compute/bandwidth
  savings (block selection — each KV block read once) "rather than KV cache
  compression", so KV memory is **not** reduced: the web sizes M3 as a full
  GQA KV cache plus a small per-token block-selection indexer key cache.

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
