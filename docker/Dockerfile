# syntax=docker/dockerfile:1
# ============================================================
# Multi-Model IDS — container image
#
# Mirrors the conda environment (Python 3.10 + requirements.txt)
# so a run is reproducible on any machine. GPU works when the host
# has the NVIDIA Container Toolkit and the container is started
# with GPU access (see docker-compose.yml).
# ============================================================
FROM python:3.10-slim

# Runtime libraries that some scientific wheels need
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgomp1 graphviz curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project and install it as an editable package
COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/workspace
ENV TF_CPP_MIN_LOG_LEVEL=2

# Streamlit 8501 | JupyterLab 8888 | MLflow 5000 | TensorBoard 6006
EXPOSE 8501 8888 5000 6006

CMD ["bash"]
