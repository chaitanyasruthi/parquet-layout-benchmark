# Reproducible environment for the Parquet layout benchmark. No cloud services needed.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# make is used as the task runner (Makefile)
RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY Makefile pytest.ini ./
COPY src ./src
COPY tests ./tests
RUN mkdir -p data/raw data/layouts results \
 && useradd --create-home bench && chown -R bench:bench /app
USER bench

# Healthy = the pinned libraries import and DuckDB can execute a query.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import duckdb, pyarrow, psutil; assert duckdb.sql('select 42').fetchone()[0] == 42"

# Default: run the whole pipeline. Override ROWS, e.g. -e ROWS=5000000
CMD ["make", "bench"]
