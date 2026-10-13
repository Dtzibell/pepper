FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13 \
    /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /opt/data-analysis/
COPY data-analysis/pyproject.toml data-analysis/uv.lock data-analysis/.python-version ./
RUN uv sync --frozen --no-install-project

COPY data-analysis/src ./src
COPY data-analysis/README.md ./README.md
RUN uv sync --frozen && \
    chmod -R a+rX /opt/data-analysis /opt/uv-python
