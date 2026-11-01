#!/bin/sh
# Builds the command line from environment variables, so the compose file is
# the only place where you have to configure anything.
#
#   STREAM          camera number, video file, or http/rtsp/tcp stream URL
#   TRIGGER         x,y,w,h of the movement trigger area
#   MODEL           path of the .tflite model
#   WEBSITE         URL of the monitoring website (optional)
#   RECORD_VIDEO    1 to record a video clip around every event
#   RECORD_OVERLAYS 1 to draw the detections on stills and video
#   PREROLL, POSTROLL  seconds of video before / after the event
#   EXTRA_ARGS      anything else, passed on as it is
set -eu

is_true() {
    case "${1:-}" in 1|true|TRUE|True|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac
}

set -- python src/catflap/main.py \
    --stream "$STREAM" \
    --trigger "$TRIGGER" \
    --model "$MODEL" \
    --record_path /app/recordings \
    --headless

if [ -n "${WEBSITE:-}" ]; then set -- "$@" --web "$WEBSITE"; fi
if is_true "${RECORD_VIDEO:-}"; then set -- "$@" --record_video; fi
if is_true "${RECORD_OVERLAYS:-}"; then set -- "$@" --record_overlays; fi
if [ -n "${PREROLL:-}" ]; then set -- "$@" --preroll "$PREROLL"; fi
if [ -n "${POSTROLL:-}" ]; then set -- "$@" --postroll "$POSTROLL"; fi

echo "Starting: $* ${EXTRA_ARGS:-}"
# EXTRA_ARGS is split into words on purpose
# shellcheck disable=SC2086
exec "$@" ${EXTRA_ARGS:-}
