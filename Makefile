.PHONY: pipeline pipeline-gpu build dev clean all test test-pipeline test-web

# Run the full data pipeline (fetch, enrich, export to JSON)
pipeline:
	cd pipeline && python -m pipeline.main --step all

# Run only the GPU pipeline (fetches from gpuhunt)
pipeline-gpu:
	cd pipeline && python -m pipeline.main --step gpu

# Install minimal Python dependencies
install-pipeline:
	cd pipeline && pip install -e .

# Install full pipeline dependencies (includes model enrichment and GPU pricing)
install-pipeline-full:
	cd pipeline && pip install -e ".[pipeline,dev]"

# Install frontend dependencies
install-web:
	cd web && npm install

# Install minimal dependencies (recommended for development)
install: install-pipeline install-web

# Install all dependencies including pipeline
install-all: install-pipeline-full install-web

# Run the frontend dev server
dev:
	cd web && npm run dev

# Build the static frontend
build:
	cd web && npm run build

# Run tests
test: test-pipeline test-web

# Run Python pipeline tests
test-pipeline:
	cd pipeline && python -m pytest tests/ -v

# Run frontend tests
test-web:
	cd web && npx vitest run

# Run Python linter
lint:
	cd pipeline && python -m ruff check .
	cd pipeline && python -m ruff format --check .

# Fix lint issues
lint-fix:
	cd pipeline && python -m ruff check --fix .
	cd pipeline && python -m ruff format .

# Full pipeline -> build workflow
all: pipeline build

# Clean build artifacts
clean:
	rm -rf web/out web/.next
	find pipeline -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
