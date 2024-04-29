#!/bin/bash

set -e
set -x

# if env does not exist
python3 -m venv env

source env/bin/activate

python -m pip list | grep amd

cd /opt/rocm-6.0.2/share/amd_smi