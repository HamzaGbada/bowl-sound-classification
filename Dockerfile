# Base image: NVIDIA PyTorch NGC container (already on host)
FROM nvcr.io/nvidia/pytorch:24.08-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync

# Copy project source
COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY results/ results/
COPY experiments/ experiments/
COPY notebooks/ notebooks/
COPY run_baselines.sh run_ablation.sh run_app.sh ./
RUN chmod +x run_baselines.sh run_ablation.sh run_app.sh

# ---------- Notebook runner ----------
FROM base AS notebooks
CMD ["uv", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# ---------- Training runner ----------
FROM base AS training
ENTRYPOINT ["bash"]
CMD ["run_baselines.sh"]

# ---------- Ablation runner ----------
FROM base AS ablation
ENTRYPOINT ["bash"]
CMD ["run_ablation.sh"]

# ---------- Streamlit app ----------
FROM base AS app
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

# ---------- Inference CLI ----------
FROM base AS inference
ENTRYPOINT ["uv", "run", "python", "app/inference.py"]