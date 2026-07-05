# Build the FASPR sidechain packer in a throwaway stage so g++/git stay out of the
# final image. FASPR finds its rotamer library (dun2010bbdep.bin) next to its binary.
FROM ubuntu:24.04 AS faspr-builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ git ca-certificates \
 && git clone --depth 1 https://github.com/tommyhuangthu/FASPR /tmp/FASPR \
 && g++ -O3 --fast-math -o /tmp/FASPR/FASPR /tmp/FASPR/src/*.cpp


FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04 AS builder

# download uv
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1

# FASPR binary + co-located rotamer library (found automatically on PATH)
COPY --from=faspr-builder /tmp/FASPR/FASPR /usr/local/bin/FASPR
COPY --from=faspr-builder /tmp/FASPR/dun2010bbdep.bin /usr/local/bin/dun2010bbdep.bin

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


