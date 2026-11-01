# syntax=docker/dockerfile:1
#
# One image for the Mac (development, tests) and the Raspberry Pi (running).
# Build for the Pi from a Mac:  docker buildx build --platform linux/arm64 -t ai-catflap .
#
# Object detection uses ai-edge-litert, which has wheels for current Python versions
# on x86_64, aarch64 (a 64-bit OS on the Pi) and the Mac.
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- build
# The full image has the compiler that RPi.GPIO needs; only the venv is copied out.
FROM python:${PYTHON_VERSION}-bookworm AS build
ARG TARGETARCH
# Pi 5 needs rpi-lgpio instead (a drop-in with the same API) - untested
ARG GPIO_LIB=RPi.GPIO
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN if [ "$TARGETARCH" = "arm64" ]; then pip install --no-cache-dir "$GPIO_LIB"; fi

# ---------------------------------------------------------------- base
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
# Shared libraries the wheels (OpenCV, LiteRT) expect from the system
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src/modules \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY src/catflap src/catflap
COPY src/modules src/modules
COPY labels.json eval_config.json trigger_config.json ./
# A default model. Normally compose mounts the one you are working on over it.
COPY test/cats.tflite models/cats.tflite
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p recordings log

# Settings are environment variables, see docker/entrypoint.sh
ENV STREAM=http://host.docker.internal:8000/stream.mjpg \
    TRIGGER=210,180,250,280 \
    DETECTOR=litert \
    MODEL=/app/models/cats.tflite

# ---------------------------------------------------------------- test
FROM base AS test
RUN pip install --no-cache-dir pytest
COPY test/ test/
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-q", "test"]

# ---------------------------------------------------------------- runtime
# Last, so a plain `docker build` makes this one
FROM base AS runtime
# The pipeline logs a statistics line every 30 s while frames arrive, so a stale log means a stuck camera or loop
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import os,sys,time; sys.exit(0 if time.time()-os.path.getmtime('log/catflap.log') < 150 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
