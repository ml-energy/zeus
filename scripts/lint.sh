#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black zeus capriccio tests
else
  black --check zeus capriccio tests
fi

ruff check zeus
pyright zeus
