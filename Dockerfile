# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && useradd --system --create-home app \
    && mkdir -p /home/app/.cache/tts-media-server/runtimes \
    && chown -R app:app /home/app/.cache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=0

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY plugins ./plugins

USER app

EXPOSE 8000

CMD ["python", "src/main.py"]
