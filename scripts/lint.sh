#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black --exclude examples/ZeusDataLoader/cifar100/models zeus capriccio examples
else
  black --check --exclude examples/ZeusDataLoader/cifar100/models zeus capriccio examples
fi

ruff zeus
