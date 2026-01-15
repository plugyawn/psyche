#!/bin/bash
# Run heterogeneous local testnet with WandB logging
#
# Each client gets its own WandB run in the same group for comparison
#
# Usage: ./scripts/run_heterogeneous_with_wandb.sh [additional args...]
#
# Required environment variables:
#   WANDB_API_KEY - Your WandB API key
#
# Optional environment variables:
#   WANDB_PROJECT - Project name (default: psyche)
#   WANDB_GROUP - Group name (default: heterogeneous-<timestamp>)
#   CONFIG_PATH - Config directory (default: ./config/nanogpt-20m-run)
#   TIERS - Comma-separated tier assignments (default: 0,1,1)
#   NUM_CLIENTS - Number of clients (default: 3)
#   EXIT_AFTER_SECS - Exit after N seconds (default: 300)

set -e

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY required"
    echo "Get your API key from: https://wandb.ai/authorize"
    exit 1
fi

# Set defaults
WANDB_PROJECT="${WANDB_PROJECT:-psyche}"
WANDB_GROUP="${WANDB_GROUP:-heterogeneous-$(date +%s)}"
CONFIG_PATH="${CONFIG_PATH:-./config/nanogpt-20m-run}"
TIERS="${TIERS:-0,1,1}"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
EXIT_AFTER_SECS="${EXIT_AFTER_SECS:-300}"

# Set library path for PyTorch (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-./.venv/lib/python3.12/site-packages/torch/lib}"
fi

# Count tiers to ensure num_clients matches
TIER_COUNT=$(echo "$TIERS" | tr ',' '\n' | wc -l | tr -d ' ')
if [ "$TIER_COUNT" != "$NUM_CLIENTS" ]; then
    echo "Warning: TIERS has $TIER_COUNT entries but NUM_CLIENTS is $NUM_CLIENTS"
    echo "Adjusting NUM_CLIENTS to match TIERS"
    NUM_CLIENTS=$TIER_COUNT
fi

# Generate helper fractions (all zeros for now)
HELPER_FRACTIONS=$(printf '0,%.0s' $(seq 1 $NUM_CLIENTS) | sed 's/,$//')

echo "=== Heterogeneous Training with WandB ==="
echo "Project: $WANDB_PROJECT"
echo "Group: $WANDB_GROUP"
echo "Config: $CONFIG_PATH"
echo "Tiers: $TIERS"
echo "Helper fractions: $HELPER_FRACTIONS"
echo "Clients: $NUM_CLIENTS"
echo "Exit after: ${EXIT_AFTER_SECS}s"
echo ""
echo "WandB Features:"
echo "  - Step-level logging: enabled"
echo "  - System metrics: enabled (5s interval)"
echo "=========================================="
echo

# Run the local testnet
exec cargo run --release -p psyche-centralized-local-testnet -- start \
    --headless \
    --headless-exit-after-secs "$EXIT_AFTER_SECS" \
    --num-clients "$NUM_CLIENTS" \
    --config-path "$CONFIG_PATH" \
    --client-matformer-tiers "$TIERS" \
    --client-matformer-helper-fractions "$HELPER_FRACTIONS" \
    --tui false \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-step-logging \
    --wandb-system-metrics \
    --wandb-system-metrics-interval-secs 5 \
    "$@"
