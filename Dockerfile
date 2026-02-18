FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir python-dotenv>=1.0 beautifulsoup4>=4.12 gpuhunt>=0.1 httpx>=0.27 pydantic>=2.7
COPY scripts/run_daily.sh /run_daily.sh
RUN chmod +x /run_daily.sh
ENTRYPOINT ["/run_daily.sh"]
