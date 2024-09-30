#!/usr/bin/env bash

# This script builds a local version of the documentation and makes it available at localhost:7777.
# By default it does not build social preview cards. If you want to debug social cards,
# set the environment variable `BUILD_SOCIAL_CARD=true` to this script.

pip list | grep mkdocs-material 2>&1 >/dev/null

if [[ ! $? -eq 0 ]]; then
  echo "Run the following to install documentation build dependnecies:"
  echo "    pip install '.[docs]'"
  exit 1
fi

mkdocs serve -a localhost:7777 -w zeus --strict
