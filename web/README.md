# Coding Model Seeks GPU — Web Frontend

Next.js static site that helps users choose the best coding LLM for their GPU setup (or vice versa). Two personas:

- **Performance**: "What GPU setup do I need for the best model?" — ranks top 3 models by benchmark score, shows GPU options per team size tier.
- **Budget**: "What's the best model for my existing GPUs?" — user picks a GPU config, sees which models fit and their cost per user.

## Tech Stack

- Next.js 16 (static export), React 19, TypeScript
- Tailwind CSS 4, shadcn/ui, Radix UI
- Vitest for testing

## Data

No database at runtime. The pipeline exports JSON files to `public/data/`:

- `gpus.json` — GPU offerings with pricing by region
- `gpu_source.json` — metadata about the GPU data source
- `models.json` — model specs (params, precision, architecture, KV cache fields)
- `snapshots/<date>/benchmarks.json` — benchmark scores and rankings per snapshot
- `snapshots/<date>/sota_scores.json` — state-of-the-art scores per benchmark per snapshot

## Key Files

| File | Purpose |
|------|---------|
| `src/app/page.tsx` | Main entry point, persona routing |
| `src/types/index.ts` | All TypeScript interfaces |
| `src/lib/calculations.ts` | Core math: memory sizing, KV cache, decode throughput, team capacity |
| `src/lib/matrix-calculator.ts` | Matrix generation for both personas |
| `src/lib/gpu-specs.ts` | GPU throughput/bandwidth specs |
| `src/lib/regime-params.ts` | Usage regime parameters (low/high concurrency, long context) |
| `src/components/recommendation-matrix.tsx` | Shared matrix table component |
| `src/components/performance-flow.tsx` | Performance persona flow |
| `src/components/budget-flow.tsx` | Budget persona flow |

## Memory Calculation

Models are sized at their native precision. Mixed-precision models (e.g. Kimi "INT4") split memory: routed experts at INT4 (0.5 B/param), attention/shared layers at BF16 (2 B/param). A 1.15x overhead factor accounts for activations and CUDA context.

## Development

```bash
npm install
npm run dev      # Dev server at localhost:3000
npm run test     # Run vitest
npm run build    # Static export
```

Or from the repo root:

```bash
make install-web
make dev
make build
```
