#!/usr/bin/env bash

# This requires uv to be installed. See https://docs.astral.sh/uv/.

set -e

uv tool run --from licensecheck licensecheck
