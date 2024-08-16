#!/usr/bin/env bash

pip list | grep mkdocs-material 2>&1 >/dev/null

if [[ ! $? -eq 0 ]]; then
  echo "Run the following to install documentation build dependnecies:"
  echo "    pip install '.[docs]'"
  exit 1
fi

mkdocs serve -a localhost:7777 -w zeus --strict
