FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04 AS builder

# download uv
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1

# set working directory
WORKDIR /app

# copy project information
COPY pyproject.toml uv.lock ./

# Install dependencies into a standard location
RUN uv python install 3.12 && uv sync --frozen --no-install-project --no-dev

# set PATH to venv
ENV PATH="/app/.venv/bin:$PATH"

# copy esmfold exec
COPY esmfold.py /usr/local/bin/esmfold
RUN chmod +x /usr/local/bin/esmfold


