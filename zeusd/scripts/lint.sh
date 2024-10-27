#!/usr/bin/env bash

set -ev

cargo fmt --all
cargo check --all
cargo clippy --all
