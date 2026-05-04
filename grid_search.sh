#!/bin/bash
# Grid search over up_ft_block and sigma for TAP-Vid DAVIS feature extraction.
#
# Usage:
#   chmod +x grid_search.sh
#   CUDA_VISIBLE_DEVICES=2 bash grid_search.sh

DATA_PATH="data/tapvid_davis/tapvid_davis.pkl"
#OUTPUT_DIR="output/tapvid_davis"
OUTPUT_DIR="/content/output"

BLOCKS=(2)

SIGMAS=(0.003)

CIN=(2.5)

ENSEMBLE_SIZE=4
MAX_FRAMES=100

TOTAL=$(( ${#BLOCKS[@]} * ${#SIGMAS[@]} * ${#CIN[@]} ))
COUNT=0

for block in "${BLOCKS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        for cin in "${CIN[@]}"; do
            COUNT=$((COUNT + 1))
            echo "========================================"
            echo "[$COUNT/$TOTAL] block=$block  sigma=$sigma cin=$cin query_mode=first"
            echo "========================================"

            python extract_features.py \
                --data_path     "$DATA_PATH" \
                --output_dir    "$OUTPUT_DIR/block${block}_sigma${sigma}_cin${cin}" \
                --up_ft_block   "$block" \
                --sigma         "$sigma" \
                --cin           "$cin" \
                --query_mode    "first"

            if [ $? -ne 0 ]; then
                echo "ERROR: block=$block sigma=$sigma cin=$cin failed, skipping..."
            fi
        done
    done
done

echo "========================================"
echo "Grid search complete. $TOTAL runs finished."
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"