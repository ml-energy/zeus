#!/bin/bash

if [[ -n "$1" ]]; then
  export PERSEUS_SCHEDULER="$1"
fi

if [[ -n "$2" ]]; then
  export PERSEUS_SCHEDULER_ARGS="$2"
fi

if [[ -n "$3" ]]; then
  export PERSEUS_MODE="$3"
fi

PORT=${PORT:-7787}

uvicorn perseus.server.main:app --host 0.0.0.0 --log-level debug --port $PORT
