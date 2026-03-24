#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "$f" ]; then
    echo "Downloading ${f}.gz ..."
    curl -sL -O "${BASE_URL}/${f}.gz"
    gunzip -f "${f}.gz"
  else
    echo "${f} already exists, skipping."
  fi
done

echo "Done. Files in ${DATA_DIR}:"
ls -lh "$DATA_DIR"
