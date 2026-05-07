# kb-core web UI image.
#
# Built locally and shipped via `docker save | docker load`. The image bakes
# in Python deps + the bge-m3 embedding model so it runs offline (important
# for Aliyun mainland where HuggingFace Hub access is flaky).
#
# Mounts at runtime:
#   /app/data/projects/<slug>/   ← project data (PDFs, structured/, db/, etc.)
#   /app/.env                    ← API keys (or use --env-file)
#
# Default command launches Streamlit on 0.0.0.0:8501. Override with
# `docker run ... <image> kb <subcommand>` to run CLI commands.

FROM python:3.12-slim

# Install system libs that some Python deps need at runtime.
# libgomp1 — for sentence-transformers / torch openmp
# libglib2.0-0 — pdfplumber/pymupdf
# curl — for healthcheck or debug
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip (works in any registry context, including mainland mirrors).
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy lockfile + project metadata first for better layer caching.
COPY pyproject.toml uv.lock .python-version ./

# Install Python deps into /app/.venv. --no-dev skips pytest etc.
RUN uv sync --frozen --no-dev

# Copy source after deps so code edits don't bust the deps layer.
COPY src ./src
COPY web ./web
COPY scripts ./scripts

# Pre-download bge-m3 into the image so first run does not need HF Hub.
# This is ~4 GB. The whole image ends up ~5-6 GB; docker-save tarball ~3 GB.
ENV HF_HOME=/app/.hf_cache
RUN uv run python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-m3', cache_folder='/app/.hf_cache/hub')"

# At runtime, prevent any HF Hub network calls (model is local).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Streamlit headless config — bind 0.0.0.0 so docker -p maps work.
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Default: launch the web UI. Override with `docker run ... kb <cmd>` for CLI.
CMD ["uv", "run", "streamlit", "run", "web/Home.py"]
