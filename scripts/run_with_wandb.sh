#!/bin/bash
# Run Psyche client with full WandB logging enabled
#
# Usage: ./scripts/run_with_wandb.sh [additional args...]
#
# Required environment variables:
#   WANDB_API_KEY - Your WandB API key
#
# Optional environment variables:
#   WANDB_PROJECT - Project name (default: psyche)
#   WANDB_RUN - Run name (default: auto-generated)
#   WANDB_GROUP - Group name for organizing runs
#   WANDB_ENTITY - WandB team/entity

set -e

# Validate WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is required"
    echo "Get your API key from: https://wandb.ai/authorize"
    exit 1
fi

# Set defaults
WANDB_PROJECT="${WANDB_PROJECT:-psyche}"
WANDB_RUN="${WANDB_RUN:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# Set library path for PyTorch (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-./.venv/lib/python3.12/site-packages/torch/lib}"
fi

# Build WandB args
WANDB_ARGS="--wandb-project $WANDB_PROJECT"
WANDB_ARGS="$WANDB_ARGS --wandb-step-logging"
WANDB_ARGS="$WANDB_ARGS --wandb-system-metrics"
WANDB_ARGS="$WANDB_ARGS --wandb-system-metrics-interval-secs 5"

[ -n "$WANDB_RUN" ] && WANDB_ARGS="$WANDB_ARGS --wandb-run $WANDB_RUN"
[ -n "$WANDB_GROUP" ] && WANDB_ARGS="$WANDB_ARGS --wandb-group $WANDB_GROUP"
[ -n "$WANDB_ENTITY" ] && WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"

echo "=== WandB Configuration ==="
echo "Project: $WANDB_PROJECT"
echo "Run: ${WANDB_RUN:-auto}"
echo "Group: ${WANDB_GROUP:-none}"
echo "Entity: ${WANDB_ENTITY:-default}"
echo "Step logging: enabled"
echo "System metrics: enabled (5s interval)"
echo "==========================="
echo

# Run the client
exec cargo run --release -p psyche-centralized-client -- train \
    $WANDB_ARGS \
    "$@"
