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
# Then strip the CUDA stack (we only do CPU embedding inference):
#   • Replace CUDA-bundled torch wheel (~884 MB) with the CPU-only wheel
#     (~200 MB) from PyTorch's CPU index.
#   • Delete the nvidia/* CUDA libs (~2.8 GB) and triton (~600 MB GPU compiler).
# Net saving: ~4 GB out of the venv.
ENV VIRTUAL_ENV=/app/.venv
RUN uv sync --frozen --no-dev \
 && uv pip uninstall torch \
 && uv pip install --no-deps --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
 && rm -rf /app/.venv/lib/python3.12/site-packages/nvidia \
           /app/.venv/lib/python3.12/site-packages/triton \
           /root/.cache/uv /tmp/*

# Copy source after deps so code edits don't bust the deps layer.
COPY src ./src
COPY web ./web
COPY scripts ./scripts

# Pre-download bge-m3 into the image so first run does not need HF Hub.
# allow_patterns excludes pytorch_model.bin (~2.2 GB duplicate of safetensors)
# and the onnx/ folder. We keep the safetensors + tokenizer + configs only.
ENV HF_HOME=/app/.hf_cache
# Use the venv python directly — `uv run` would re-sync from the lockfile and
# undo our CPU-torch swap (lockfile pins torch==2.11.0 from PyPI = CUDA build).
#
# Strategy: snapshot_download with ignore_patterns (skip pytorch_model.bin
# duplicate + ONNX variants). Then SentenceTransformer.encode() acts as a
# load-test — if any required file is missing, build fails here, not at
# runtime in production.
RUN /app/.venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-m3', cache_dir='/app/.hf_cache/hub', \
    ignore_patterns=['pytorch_model.bin', 'onnx/*', '*.onnx', \
                     'colbert_linear.pt', 'sparse_linear.pt']); \
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('BAAI/bge-m3', cache_folder='/app/.hf_cache/hub'); \
v = m.encode(['build-time check']); \
print(f'bge-m3 OK: encoded shape={v.shape}')"

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

# Make venv binaries first on PATH, so `streamlit`, `kb`, `python` resolve to
# /app/.venv/bin/* without needing `uv run` (which would auto-re-sync from
# the lockfile and undo the CPU-torch swap).
ENV PATH="/app/.venv/bin:${PATH}"

# Default: launch the web UI. Override with `docker run ... kb <cmd>` for CLI.
CMD ["streamlit", "run", "web/Home.py"]
