# Coding Model Seeks GPU

Open source coding LLMs ranked by real-world performance, sized to real hardware.

Pick a model and find the GPU setup you need, or pick a GPU and find the best model that fits. Data is refreshed daily with live cloud pricing from [gpuhunt](https://github.com/dstackai/gpuhunt) and benchmark scores from the [OpenHands Index](https://index.openhands.dev).

## How it works

A **Python pipeline** fetches GPU pricing across 13+ cloud providers, enriches model specs from HuggingFace, and exports everything as static JSON. A **Next.js frontend** loads that JSON and lets users explore model-GPU pairings through three lenses:

- **Performance** — "What GPU do I need for the best model?" Ranks top models by benchmark score, shows GPU options at different concurrency tiers with monthly costs.
- **Budget** — "What's the best model for my GPU?" Select a GPU config and see which models fit, with cost breakdowns.
- **Trends** — Historical benchmark and pricing data over time.

The app calculates VRAM requirements (including KV cache for MLA and GQA attention), throughput estimates, and concurrency capacity for each model-GPU combination.

## Project structure

```
.
├── pipeline/                  # Python data pipeline
│   ├── pipeline/
│   │   ├── main.py            # CLI entry point
│   │   ├── config.py          # Paths and env config
│   │   ├── enrichment.py      # ModelSpec dataclass
│   │   ├── sources/           # Data fetchers (gpuhunt, HuggingFace)
│   │   ├── exporters/         # JSON export to web/public/data/
│   │   └── snapshots/         # Historical benchmark snapshots
│   └── tests/
├── web/                       # Next.js frontend
│   ├── src/
│   │   ├── app/page.tsx       # Main page
│   │   ├── components/        # UI components
│   │   ├── lib/               # Calculations, data loading, GPU specs
│   │   └── types/             # TypeScript interfaces
│   └── public/data/           # Pipeline output (JSON)
├── external/                  # Git submodules
│   └── openhands-index-results/
├── scripts/run_daily.sh       # Automated daily pipeline
├── Dockerfile                 # Pipeline runner image
├── Makefile                   # Dev commands
└── .github/workflows/         # CI/CD (GitHub Pages deploy)
```

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 22+

### Install

```bash
# Install everything
make install-all

# Or install separately
make install-pipeline    # Python pipeline (minimal deps)
make install-web         # Frontend (npm install)
```

### Development

```bash
# Start the dev server
make dev

# Run the full pipeline (fetch GPUs, enrich models, generate snapshots, export)
make pipeline

# Run only the GPU pricing step
make pipeline-gpu

# Full pipeline then build
make all
```

### Build

```bash
make build    # Static export to web/out/
```

### Test & lint

```bash
make test       # Pipeline tests (pytest)
make lint       # Ruff check + format check
make lint-fix   # Auto-fix
cd web && npm test   # Frontend tests (vitest)
```

## Data flow

```
gpuhunt (13+ cloud providers)  ──►  pipeline  ──►  web/public/data/gpus.json
HuggingFace API                ──►  pipeline  ──►  web/public/data/models.json
OpenHands Index (submodule)    ──►  pipeline  ──►  web/public/data/snapshots/
                                                   web/public/data/gpu_source.json
```

The pipeline writes static JSON to `web/public/data/`. The Next.js app loads these files at runtime. A daily GitHub Actions workflow runs the pipeline in Docker, commits any changes, and triggers a GitHub Pages deploy.

## Environment variables

| Variable | Where | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_ENABLE_LOCATION_FILTER` | Web (build-time) | unset | Set to `"true"` to show the region picker and filter GPUs by location. When unset, all regions are shown with results deduplicated to the cheapest per GPU config. |

## Deployment

The site deploys automatically to GitHub Pages on every push to `main` via `.github/workflows/deploy-pages.yml`. The daily pipeline runs in a Docker container (`Dockerfile` + `scripts/run_daily.sh`) that clones the repo, updates data, and pushes changes back to `main`.

## License

See repository for license details.
