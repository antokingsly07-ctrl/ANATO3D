# ANATO3D backend — CPU-only image.
#
# The optional MONAI/torch AI path is deliberately NOT installed here: it is
# large and needs a user-supplied model bundle. Extend this Dockerfile if you
# want to enable it (and set ANATO3D_AI_MODEL_PATH).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is used only by the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY . .

# All working data lives under /data so it can be mounted as a disposable volume.
ENV ANATO3D_DATA_DIR=/data \
    ANATO3D_HOST=0.0.0.0 \
    ANATO3D_PORT=8000 \
    ANATO3D_DEBUG=false

RUN mkdir -p /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
