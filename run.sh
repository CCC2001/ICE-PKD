#!/bin/bash
set -euo pipefail

# 制表符分隔：train_file  test_file
while IFS=$'\t' read -r train_file test_file; do
    echo "[$(date)] Run incremental training: ${train_file}  ->  ${test_file}"
    python main.py \
        --train_path "/nfs/volume-661-1/chenbaining/data/data_2d/data_3/${train_file}" \
        --test_path  "/nfs/volume-661-1/chenbaining/data/data_2d/data_3/${test_file}"
done < date.txt