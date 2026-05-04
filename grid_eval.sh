#!/bin/bash
# Run evaluation for all (block, sigma) combinations produced by grid search.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash evaluate_grid.sh

BLOCKS=(2)

BLOCKS=(2)

SIGMAS=(0.003)

CIN=(2.5)

FEAT_BASE="/content/output"
RESULT_BASE="/content/result"

TOTAL=$(( ${#BLOCKS[@]} * ${#SIGMAS[@]} * ${#CIN[@]} ))
COUNT=0

for block in "${BLOCKS[@]}"; do
    for sigma in "${SIGMAS[@]}"; do
        for cin in "${CIN[@]}"; do
            COUNT=$((COUNT + 1))

            FEAT_DIR="${FEAT_BASE}/block${block}_sigma${sigma}_cin${cin}"
            OUTPUT_DIR="${RESULT_BASE}/block${block}_sigma${sigma}_cin${cin}"

            # skip if feature dir does not exist
            if [ ! -d "$FEAT_DIR" ]; then
                echo "[$COUNT/$TOTAL] SKIP (not found): $FEAT_DIR"
                continue
            fi

            echo "========================================"
            echo "[$COUNT/$TOTAL] block=$block  sigma=$sigma"
            echo "  feat_dir:   $FEAT_DIR"
            echo "  output_dir: $OUTPUT_DIR"
            echo "========================================"

            python evaluate.py \
                --feat_dir   "$FEAT_DIR" \
                --output_dir "$OUTPUT_DIR"

            if [ $? -ne 0 ]; then
                echo "ERROR: block=$block sigma=$sigma failed, skipping..."
            fi
        done
    done
done

echo "========================================"
echo "All evaluations complete."
echo "========================================"