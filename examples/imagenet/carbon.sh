#!/bin/bash

export ZEUS_TARGET_METRIC="0.60"               # Stop training when target val metric is reached
export ZEUS_LOG_DIR="carbon_zeus_log"                 # Directory to store profiling logs
export ZEUS_JOB_ID="zeus"                      # Used to distinguish recurrences, so not important
export ZEUS_COST_THRESH="inf"                  # Kill training when cost (Equation 2) exceeds this
export ZEUS_ETA_KNOB="0.25"                     # Knob to tradeoff energy and time (Equation 2)
export ZEUS_MONITOR_PATH="/workspace/zeus/zeus_monitor/zeus_monitor" # Path to power monitor
export ZEUS_PROFILE_PARAMS="20,80"             # warmup_iters,profile_iters for each power limit
export ZEUS_USE_OPTIMAL_PL="True"              # Whether to acutally use the optimal PL found

export ZEUS_USE_CARBON="True"                  # Whether to acutally use the carbon-to-accuracy (CTA) instead of ETA

# Single-GPU
python train.py \
    /data/imagenet/ \
    --gpu 0                 `# Specify the GPU id to use` \
    --zeus \
    --batch_size 128 \
    --seed 1 \
    --arch resnet50 \
