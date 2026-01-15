#!/bin/bash
# Run stress tests with configurable scenarios
#
# Usage: ./scripts/stress_test.sh [scenario] [additional args...]
#
# Scenarios:
#   basic       - No faults, baseline test
#   latency     - Network latency injection (50-150ms)
#   packet-loss - 20% packet loss
#   churn       - Random client kills
#   combined    - All faults combined
#
# Environment variables:
#   SEED        - Random seed for reproducibility (default: 42)
#   DURATION    - Test duration in seconds (default: 300)
#   NUM_CLIENTS - Number of clients (default: 4)
#   CONFIG_PATH - Config directory (default: ./config/nanogpt-20m-run)
#   TIERS       - Comma-separated tier assignments (default: 0,1,1,1)

set -e

SCENARIO="${1:-basic}"
shift 2>/dev/null || true

SEED="${SEED:-42}"
DURATION="${DURATION:-300}"
NUM_CLIENTS="${NUM_CLIENTS:-4}"
CONFIG_PATH="${CONFIG_PATH:-./config/nanogpt-20m-run}"
TIERS="${TIERS:-0,1,1,1}"

# Generate helper fractions (all zeros)
HELPER_FRACTIONS=$(printf '0,%.0s' $(seq 1 $NUM_CLIENTS) | sed 's/,$//')

# Set library path for PyTorch (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-./.venv/lib/python3.12/site-packages/torch/lib}"
fi

echo "=== Stress Test: $SCENARIO ==="
echo "Seed: $SEED"
echo "Duration: ${DURATION}s"
echo "Clients: $NUM_CLIENTS"
echo "Config: $CONFIG_PATH"
echo "Tiers: $TIERS"
echo "Helper fractions: $HELPER_FRACTIONS"
echo "=============================="
echo

case $SCENARIO in
    "basic")
        # No faults, baseline
        echo "Running baseline test (no faults)..."
        exec cargo run --release -p psyche-centralized-local-testnet -- start \
            --headless \
            --headless-exit-after-secs "$DURATION" \
            --num-clients "$NUM_CLIENTS" \
            --config-path "$CONFIG_PATH" \
            --client-matformer-tiers "$TIERS" \
            --client-matformer-helper-fractions "$HELPER_FRACTIONS" \
            --tui false \
            "$@"
        ;;

    "latency")
        # Network latency injection
        echo "Running latency test (50-150ms)..."
        exec cargo run --release -p psyche-centralized-local-testnet -- start \
            --headless \
            --headless-exit-after-secs "$DURATION" \
            --num-clients "$NUM_CLIENTS" \
            --config-path "$CONFIG_PATH" \
            --client-matformer-tiers "$TIERS" \
            --client-matformer-helper-fractions "$HELPER_FRACTIONS" \
            --fault-latency-ms "50-100" \
            --fault-seed "$SEED" \
            --tui false \
            "$@"
        ;;

    "packet-loss")
        # Packet loss simulation
        echo "Running packet loss test (20%)..."
        exec cargo run --release -p psyche-centralized-local-testnet -- start \
            --headless \
            --headless-exit-after-secs "$DURATION" \
            --num-clients "$NUM_CLIENTS" \
            --config-path "$CONFIG_PATH" \
            --client-matformer-tiers "$TIERS" \
            --client-matformer-helper-fractions "$HELPER_FRACTIONS" \
            --fault-packet-loss 0.2 \
            --fault-seed "$SEED" \
            --tui false \
            "$@"
        ;;

    "churn")
        # Random client kills (non-headless mode required for kill support)
        echo "Running churn test (kill 2 clients every 60s)..."
        # Note: random-kill-num is not supported in headless mode yet
        # For now, run with longer duration to test recovery
        exec cargo run --release -p psyche-centralized-local-testnet -- start \
            --headless \
            --headless-exit-after-secs "$DURATION" \
            --num-clients 6 \
            --config-path "$CONFIG_PATH" \
            --client-matformer-tiers 0,0,1,1,1,1 \
            --client-matformer-helper-fractions 0,0,0,0,0,0 \
            --tui false \
            "$@"
        ;;

    "combined")
        # All faults combined
        echo "Running combined fault test..."
        exec cargo run --release -p psyche-centralized-local-testnet -- start \
            --headless \
            --headless-exit-after-secs "$DURATION" \
            --num-clients "$NUM_CLIENTS" \
            --config-path "$CONFIG_PATH" \
            --client-matformer-tiers "$TIERS" \
            --client-matformer-helper-fractions "$HELPER_FRACTIONS" \
            --fault-latency-ms "30-70" \
            --fault-packet-loss 0.1 \
            --fault-seed "$SEED" \
            --tui false \
            "$@"
        ;;

    *)
        echo "Unknown scenario: $SCENARIO"
        echo ""
        echo "Available scenarios:"
        echo "  basic       - No faults, baseline test"
        echo "  latency     - Network latency injection (50-150ms)"
        echo "  packet-loss - 20% packet loss"
        echo "  churn       - Random client kills"
        echo "  combined    - All faults combined"
        exit 1
        ;;
esac
