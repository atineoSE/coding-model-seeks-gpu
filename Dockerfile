FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pipeline/pyproject.toml /tmp/pipeline/pyproject.toml
RUN pip install --no-cache-dir /tmp/pipeline[pipeline] && rm -rf /tmp/pipeline
COPY scripts/run_daily.sh /run_daily.sh
RUN chmod +x /run_daily.sh
ENTRYPOINT ["/run_daily.sh"]
