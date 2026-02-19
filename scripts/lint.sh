#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format zeus tests
else
  ruff format --check zeus tests
fi

ruff check zeus
ty check zeus
