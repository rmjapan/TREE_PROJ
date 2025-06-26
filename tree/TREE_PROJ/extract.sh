#!/bin/bash

REMOTE_DIR="/home/ryuichi/tree/TREE_PROJ/l-strings"
cd "$REMOTE_DIR"

for FILE in *.tar.gz; do
  tar -xzf "$FILE"
  rm "$FILE"
done
