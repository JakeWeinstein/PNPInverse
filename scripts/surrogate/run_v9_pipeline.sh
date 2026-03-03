#!/usr/bin/env bash
# ===========================================================================
# v9 Surrogate Pipeline
# ===========================================================================
#
# Full pipeline: generate training data -> build surrogate -> run inference.
#
# Usage (from PNPInverse/ directory):
#   bash scripts/surrogate/run_v9_pipeline.sh
#
# Or run individual steps:
#   bash scripts/surrogate/run_v9_pipeline.sh generate
#   bash scripts/surrogate/run_v9_pipeline.sh build
#   bash scripts/surrogate/run_v9_pipeline.sh infer
#
# Environment:
#   PYTHON  - path to python binary (default: auto-detect venv-firedrake)
#   WORKERS - number of PDE workers for inference phase 3 (default: 0 = auto)
#
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PNPINVERSE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Auto-detect python if not set
if [ -z "${PYTHON:-}" ]; then
    PYTHON="$PNPINVERSE_ROOT/../venv-firedrake/bin/python"
    if [ ! -x "$PYTHON" ]; then
        PYTHON="$(which python3 2>/dev/null || which python 2>/dev/null)"
    fi
fi

WORKERS="${WORKERS:-0}"

# Output paths
OUTPUT_DIR="$PNPINVERSE_ROOT/StudyResults/surrogate_v9"
TRAINING_DATA="$OUTPUT_DIR/training_data_500.npz"
SURROGATE_MODEL="$OUTPUT_DIR/surrogate_model.pkl"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "======================================================================"
echo "  v9 SURROGATE PIPELINE"
echo "  Python:   $PYTHON"
echo "  Root:     $PNPINVERSE_ROOT"
echo "  Output:   $OUTPUT_DIR"
echo "  Workers:  $WORKERS"
echo "  Date:     $(date)"
echo "======================================================================"

# Determine which step to run (default: all)
STEP="${1:-all}"

# ---------------------------------------------------------------------------
# Step 1: Generate training data (N=500: 400 wide + 100 focused)
# ---------------------------------------------------------------------------
run_generate() {
    echo ""
    echo ">>> STEP 1: Generate training data (400 wide + 100 focused = 500 total)"
    echo "    Output: $TRAINING_DATA"
    echo "    Log:    $LOG_DIR/generate.log"
    echo "    This may take several hours..."
    echo ""

    cd "$PNPINVERSE_ROOT"
    "$PYTHON" scripts/surrogate/generate_training_data.py \
        --n-samples 400 \
        --n-focused 100 \
        --seed 42 \
        --focused-seed 99 \
        --checkpoint-interval 5 \
        --min-converged 0.8 \
        --output "$TRAINING_DATA" \
        2>&1 | tee "$LOG_DIR/generate.log"

    echo ""
    echo ">>> STEP 1 COMPLETE: Training data saved to $TRAINING_DATA"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 2: Build surrogate model (with cross-validation for PC smoothing)
# ---------------------------------------------------------------------------
run_build() {
    echo ""
    echo ">>> STEP 2: Build surrogate model (with cross-validation)"
    echo "    Input:  $TRAINING_DATA"
    echo "    Output: $SURROGATE_MODEL"
    echo "    Log:    $LOG_DIR/build.log"
    echo ""

    if [ ! -f "$TRAINING_DATA" ]; then
        echo "ERROR: Training data not found at $TRAINING_DATA"
        echo "       Run 'bash scripts/surrogate/run_v9_pipeline.sh generate' first."
        exit 1
    fi

    cd "$PNPINVERSE_ROOT"
    "$PYTHON" scripts/surrogate/build_surrogate.py \
        --training-data "$TRAINING_DATA" \
        --output "$SURROGATE_MODEL" \
        --cross-validate-smoothing \
        --test-fraction 0.1 \
        --validation-split 0.1 \
        --kernel thin_plate_spline \
        --degree 1 \
        --smoothing 0.0 \
        2>&1 | tee "$LOG_DIR/build.log"

    echo ""
    echo ">>> STEP 2 COMPLETE: Surrogate model saved to $SURROGATE_MODEL"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 3: Run v9 inference (surrogate + PDE refinement)
# ---------------------------------------------------------------------------
run_infer() {
    echo ""
    echo ">>> STEP 3: Run v9 inference (surrogate + PDE refinement)"
    echo "    Model:  $SURROGATE_MODEL"
    echo "    Log:    $LOG_DIR/infer.log"
    echo ""

    if [ ! -f "$SURROGATE_MODEL" ]; then
        echo "ERROR: Surrogate model not found at $SURROGATE_MODEL"
        echo "       Run 'bash scripts/surrogate/run_v9_pipeline.sh build' first."
        exit 1
    fi

    cd "$PNPINVERSE_ROOT"
    "$PYTHON" scripts/surrogate/Infer_BVMaster_charged_v9_surrogate.py \
        --model "$SURROGATE_MODEL" \
        --workers "$WORKERS" \
        --pde-maxiter 10 \
        2>&1 | tee "$LOG_DIR/infer.log"

    echo ""
    echo ">>> STEP 3 COMPLETE: Inference results in $OUTPUT_DIR/"
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$STEP" in
    generate)
        run_generate
        ;;
    build)
        run_build
        ;;
    infer)
        run_infer
        ;;
    all)
        run_generate
        run_build
        run_infer
        echo "======================================================================"
        echo "  v9 PIPELINE COMPLETE"
        echo "  Results: $OUTPUT_DIR/"
        echo "  Date:    $(date)"
        echo "======================================================================"
        ;;
    *)
        echo "Usage: $0 {generate|build|infer|all}"
        exit 1
        ;;
esac
