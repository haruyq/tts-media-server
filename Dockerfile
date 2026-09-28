# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg g++ git \
    && useradd --system --create-home app \
    && mkdir -p /home/app/.cache/tts-media-server/runtimes \
    && chown -R app:app /home/app/.cache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# プラグインのruntimeを再構築する際にtorch等を再取得しないよう、uvのキャッシュを
# runtimeと同じボリュームへ置く。同じファイルシステム上ではハードリンクで展開される
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/home/app/.cache/tts-media-server/runtimes/.uv-cache \
    UV_PYTHON_DOWNLOADS=0

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY src ./src
COPY plugins ./plugins

USER app

EXPOSE 8000

CMD ["python", "src/main.py"]
