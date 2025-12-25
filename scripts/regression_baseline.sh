#!/usr/bin/env bash
# Regression Baseline Management Script
#
# This script helps establish and validate training regression baselines.
# Use it to create reproducible baselines and verify that code changes
# don't introduce silent convergence regressions.
#
# Usage:
#   ./scripts/regression_baseline.sh establish  - Create a new baseline
#   ./scripts/regression_baseline.sh validate   - Validate against existing baseline
#   ./scripts/regression_baseline.sh compare <baseline> <current>  - Compare two metrics files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
BASELINE_DIR="${ROOT_DIR}/baselines"
SEED=42
STEPS=100
MODEL="checkpoints/tiny-llama-local"
DATA="data/tinyshakespeare-bin"
TOLERANCE="0.01"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Ensure required directories exist
ensure_dirs() {
    mkdir -p "$BASELINE_DIR"
}

# Check if prerequisites are met
check_prerequisites() {
    if [[ ! -d "${ROOT_DIR}/${MODEL}" ]]; then
        log_error "Model checkpoint not found at ${MODEL}"
        log_info "Run 'python scripts/prepare_tinyshakespeare_bin_dataset.py' first"
        exit 1
    fi

    if [[ ! -d "${ROOT_DIR}/${DATA}" ]]; then
        log_error "Training data not found at ${DATA}"
        log_info "Run 'python scripts/prepare_tinyshakespeare_bin_dataset.py' first"
        exit 1
    fi
}

# Run a training session and record metrics
run_training() {
    local output_file="$1"
    local description="$2"

    log_info "Running training: $description"
    log_info "Output: $output_file"

    cd "$ROOT_DIR"

    # Source environment if available
    if [[ -f scripts/psyche-env.sh ]]; then
        # Use subshell to avoid script errors affecting main script
        LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
        cargo run --release --example train -p psyche-modeling -- \
            --model "${ROOT_DIR}/${MODEL}" \
            --data-path "${ROOT_DIR}/${DATA}" \
            --sequence-length 64 \
            --micro-batch 2 \
            --total-batch 4 \
            --learning-rate 4e-4 \
            --warmup-steps 10 \
            --total-steps "$STEPS" \
            --seed "$SEED" \
            --metrics-output "$output_file" \
            --max-grad-norm 1.0 \
            2>&1 | tee "${output_file%.jsonl}.log"
    else
        log_error "psyche-env.sh not found"
        exit 1
    fi

    log_info "Training complete, metrics saved to: $output_file"
}

# Establish a new baseline
cmd_establish() {
    ensure_dirs
    check_prerequisites

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local baseline_file="${BASELINE_DIR}/baseline_${timestamp}.jsonl"
    local latest_link="${BASELINE_DIR}/latest.jsonl"

    log_info "Establishing new baseline..."
    log_info "Timestamp: $timestamp"
    log_info "Seed: $SEED"
    log_info "Steps: $STEPS"

    run_training "$baseline_file" "baseline establishment"

    # Update latest symlink
    rm -f "$latest_link"
    ln -s "$(basename "$baseline_file")" "$latest_link"

    log_info "Baseline established: $baseline_file"
    log_info "Latest symlink updated: $latest_link"

    # Show summary
    local final_loss=$(tail -1 "$baseline_file" | jq -r '.loss')
    local final_grad_norm=$(tail -1 "$baseline_file" | jq -r '.global_grad_norm')
    log_info "Final loss: $final_loss"
    log_info "Final grad_norm: $final_grad_norm"
}

# Validate against existing baseline
cmd_validate() {
    ensure_dirs
    check_prerequisites

    local baseline_file="${BASELINE_DIR}/latest.jsonl"

    if [[ ! -f "$baseline_file" ]]; then
        log_error "No baseline found at $baseline_file"
        log_info "Run './scripts/regression_baseline.sh establish' first"
        exit 1
    fi

    local current_file="${BASELINE_DIR}/current_$(date +%Y%m%d_%H%M%S).jsonl"

    log_info "Validating against baseline: $baseline_file"

    run_training "$current_file" "validation run"

    log_info "Comparing metrics..."

    cd "$ROOT_DIR"
    LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
    cargo run --release --example compare_metrics -p psyche-modeling -- \
        --baseline "$baseline_file" \
        --current "$current_file" \
        --tolerance "$TOLERANCE"

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_info "Validation PASSED"
        rm -f "$current_file"  # Clean up on success
    else
        log_error "Validation FAILED"
        log_info "Current metrics kept at: $current_file"
    fi

    return $exit_code
}

# Compare two arbitrary metrics files
cmd_compare() {
    local baseline="$1"
    local current="$2"

    if [[ ! -f "$baseline" ]]; then
        log_error "Baseline file not found: $baseline"
        exit 1
    fi

    if [[ ! -f "$current" ]]; then
        log_error "Current file not found: $current"
        exit 1
    fi

    cd "$ROOT_DIR"
    LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
    cargo run --release --example compare_metrics -p psyche-modeling -- \
        --baseline "$baseline" \
        --current "$current" \
        --tolerance "$TOLERANCE"
}

# Show help
cmd_help() {
    cat <<EOF
Regression Baseline Management Script

Usage:
    $0 establish              Create a new baseline
    $0 validate               Validate current code against latest baseline
    $0 compare <base> <curr>  Compare two metrics files
    $0 help                   Show this help

Environment Variables:
    SEED        Random seed for reproducibility (default: 42)
    STEPS       Number of training steps (default: 100)
    TOLERANCE   Maximum allowed loss deviation (default: 0.01)

Examples:
    # Create a new baseline
    $0 establish

    # Validate against existing baseline
    $0 validate

    # Compare two specific files
    $0 compare baselines/v1.jsonl baselines/v2.jsonl

    # Use custom settings
    STEPS=200 TOLERANCE=0.005 $0 establish
EOF
}

# Main dispatch
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        establish)
            cmd_establish
            ;;
        validate)
            cmd_validate
            ;;
        compare)
            if [[ $# -lt 2 ]]; then
                log_error "compare requires two arguments: <baseline> <current>"
                exit 1
            fi
            cmd_compare "$1" "$2"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            log_error "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
