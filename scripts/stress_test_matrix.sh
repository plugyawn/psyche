#!/bin/bash
# Run all stress test scenarios and collect results
#
# Usage: ./scripts/stress_test_matrix.sh
#
# Environment variables:
#   DURATION    - Test duration per scenario in seconds (default: 180)
#   SEEDS       - Comma-separated seeds to use (default: 42,123,456)
#   SCENARIOS   - Comma-separated scenarios to run (default: basic,latency,packet-loss,combined)

set -e

DURATION="${DURATION:-180}"
SEEDS="${SEEDS:-42,123,456}"
SCENARIOS="${SCENARIOS:-basic,latency,packet-loss,combined}"

# Create results directory
RESULTS_DIR="./stress_test_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== Stress Test Matrix ==="
echo "Results: $RESULTS_DIR"
echo "Duration: ${DURATION}s per scenario"
echo "Seeds: $SEEDS"
echo "Scenarios: $SCENARIOS"
echo "=========================="
echo

# Set library path for PyTorch (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-./.venv/lib/python3.12/site-packages/torch/lib}"
fi

IFS=',' read -ra SCENARIO_LIST <<< "$SCENARIOS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

TOTAL_RUNS=$((${#SCENARIO_LIST[@]} * ${#SEED_LIST[@]}))
CURRENT_RUN=0

for scenario in "${SCENARIO_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        LOG_FILE="$RESULTS_DIR/${scenario}_seed${seed}.log"

        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running $scenario with seed $seed..."
        echo "  Log: $LOG_FILE"

        SEED=$seed DURATION=$DURATION ./scripts/stress_test.sh "$scenario" \
            2>&1 | tee "$LOG_FILE"

        echo "  Completed!"
    done
done

echo ""
echo "=== Generating Summary ==="

# Generate summary
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "=== STRESS TEST SUMMARY ===" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "Duration per test: ${DURATION}s" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for log in "$RESULTS_DIR"/*.log; do
    name=$(basename "$log" .log)

    # Extract final loss (look for various loss patterns)
    final_loss=$(grep -oE 'loss[=: ]+[0-9]+\.[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "N/A")

    # Count errors/panics
    errors=$(grep -ci 'error\|panic\|failed' "$log" 2>/dev/null || echo 0)

    # Check for NaN/Inf
    nan_inf=$(grep -ci 'nan\|inf' "$log" 2>/dev/null || echo 0)

    # Extract step count if available
    steps=$(grep -oE 'step[=: ]+[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+' || echo "N/A")

    echo "$name:" >> "$SUMMARY_FILE"
    echo "  Final loss: $final_loss" >> "$SUMMARY_FILE"
    echo "  Final step: $steps" >> "$SUMMARY_FILE"
    echo "  Errors: $errors" >> "$SUMMARY_FILE"
    echo "  NaN/Inf: $nan_inf" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "=== PASS/FAIL Criteria ===" >> "$SUMMARY_FILE"
echo "basic: Loss decreases, no errors" >> "$SUMMARY_FILE"
echo "latency: Loss within 10% of basic, no timeouts" >> "$SUMMARY_FILE"
echo "packet-loss: Retries succeed, loss within 15% of basic" >> "$SUMMARY_FILE"
echo "combined: All above, plus no cascading failures" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo ""
echo "Full results saved to: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"
