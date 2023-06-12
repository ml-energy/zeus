#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black --exclude examples/cifar100/models zeus capriccio examples
else
  black --check --exclude examples/cifar100/models zeus capriccio examples
fi

ruff zeus
