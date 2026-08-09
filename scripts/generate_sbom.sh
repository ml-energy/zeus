#!/usr/bin/env bash

# This script requires uv to be installed. See https://docs.astral.sh/uv/.

set -euo pipefail

uv sync
uv pip freeze | uv tool run \
    --from cyclonedx-bom \
    cyclonedx-py requirements \
    --pyproject pyproject.toml \
    --of JSON \
    -o zeus-sbom.json \
    -
