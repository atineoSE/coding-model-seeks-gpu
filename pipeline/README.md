# Pipeline

Data pipeline that fetches GPU pricing, enriches model specs, and generates benchmark snapshots. Runs daily via Docker (`scripts/run_daily.sh`).

## Steps

1. **GPU prices** — Queries [gpuhunt](https://github.com/dstackai/gpuhunt) for NVIDIA on-demand offerings across 13+ cloud providers. Filters out consumer/low-VRAM GPUs, keeps the cheapest per `(gpu_name, vram, gpu_count)`. Exports to `gpus.json`.
2. **Benchmark snapshots** — Reads the [OpenHands Index](https://index.openhands.dev) git submodule history, generates per-date snapshots with scores, rankings, and SOTA entries. Exports to `snapshots/`.
3. **Model enrichment** — Fetches HuggingFace configs for each benchmarked model: parameter counts, architecture (Dense/MoE), context length, precision, attention type, KV cache fields. Exports to `models.json`.
4. **Export** — Writes all JSON to `web/public/data/`.
5. **API pricing** — Picks the best closed model per lab (Anthropic/OpenAI/Google) from the latest snapshot, fetches per-token pricing from [LiteLLM](https://github.com/BerriAI/litellm), and exports `api_pricing.json` (drives the API-vs-self-hosting chart).

## Overriding the closed model per lab

Best-in-lab selection is normally score-derived. To pin a different model, or to surface a closed model **before** it has OpenHands Index scores, edit `CLOSED_MODEL_OVERRIDES` (keyed by lab → `model_name`) in `pipeline/pipeline/sources/litellm_source.py`:

```python
CLOSED_MODEL_OVERRIDES = { "anthropic": "claude-opus-4-8" }
```

The value must resolve to a valid LiteLLM key via `model_name.lower()` → `+ "-preview"` → `LITELLM_ID_MAP`. Verify a candidate against the [LiteLLM price file](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json) before adding it (pricing still comes live from LiteLLM; a required lab that can't resolve fails the pipeline). Mirror the same entry in `web/src/lib/snapshot-matrix.ts` (`CLOSED_MODEL_OVERRIDES`) to keep the Snapshot Coverage Matrix consistent — an injected model with no scores yet shows there as an unranked closed row.

## Daily run flow

```
run_daily.sh
├── Clone repo + submodules
├── Update openhands-index submodule to latest
├── Upgrade gpuhunt to latest (best-effort)
├── Install pipeline
├── Run all steps (always queries gpuhunt, regardless of dependency update)
├── Check git diff on data files
│   ├── No changes → exit
│   └── Changes → commit + push
```

## Email alerts

All alerts use the `[coding-model-seeks-gpu]` subject prefix. Sent via Gmail SMTP, configured through env vars (`SMTP_USER`, `SMTP_PASSWORD`, `NOTIFY_TO`). Disabled when any var is missing. Email failures are logged but never crash the pipeline.

| Alert | Subject | When |
|-------|---------|------|
| **Data updated** | Source data updated | Pipeline succeeded with data changes |
| **Pipeline failed** | Pipeline failed | All 3 retry attempts exhausted |
| **Missing mapping** | Missing HuggingFace mapping for {model} | Benchmark model has no HF repo entry |
| **Format break** | Data format breaking change | gpuhunt API or openhands-index schema changed; skips retries, exits immediately |

## Usage

```bash
# Full pipeline
python -m pipeline.main --step all

# Single step
python -m pipeline.main --step gpu
python -m pipeline.main --step snapshots
python -m pipeline.main --step models

# Force full snapshot rebuild
python -m pipeline.main --step all --force-snapshots
```
