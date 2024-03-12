#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black zeus capriccio
else
  black zeus capriccio
fi

ruff zeus
