#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black zeus tests
else
  black --check zeus tests
fi

ruff check zeus
pyright zeus
