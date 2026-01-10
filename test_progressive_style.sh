#!/bin/bash

# Quick Test Script for Progressive Style Guidance
# Conservative parameters for testing the new system

# Model and hardware settings
INIT_MODE=2  # Simple mode for testing
GPU_IDS="3 4"
MODEL_PATH="/mnt/netdisk2/zhangyf/model/FLUX.1-dev"
OUT_DIR="results/test/quick_progressive_test"

# Small images for quick testing
HEIGHT=512
WIDTH=512
SEED=42

# Very conservative style guidance for stable testing
USE_CFG_GUIDANCE=true
CFG_GUIDANCE_SCALE=0.1           # Very gentle influence
CFG_GUIDANCE_START_SIGMA=0.9     # Start when noise is very high
CFG_GUIDANCE_END_SIGMA=0.5       # End early to avoid issues

# Debug output to see the process
DEBUG_OUTPUT_DIR="${OUT_DIR}/debug"

# Simple test prompts
BG_PROMPTS=(
    "in a colorful theme park, roller coasters and amusement rides in the background,"
    "in an arcade, flashing lights and game machines in the background,"
    "in a fantasy-themed park, castles and fairy tale characters in the background,"
)
FG_PROMPT="a American girl with long brown hair, wearing a blue dress and a white hat and a Chinese girl with long black hair, wearing a red dress and a blue hat, "
ACT_PROMPTS=(
    "riding a roller coaster, excited expression, front view"
    "playing an arcade game, focused expression, side view"
    "posing with a costumed character, happy expression, front view"
)
# Create temporary prompts file
TEMP_PROMPTS="/tmp/quick_test_prompts_$$.txt"
> "$TEMP_PROMPTS"

# Create debug directory
mkdir -p "$DEBUG_OUTPUT_DIR"

# Build prompts file
for i in "${!BG_PROMPTS[@]}"; do
    echo "${BG_PROMPTS[$i]}#${FG_PROMPT}#${ACT_PROMPTS[$i]}" >> "$TEMP_PROMPTS"
done

echo "=========================================="
echo "Quick Progressive Style Guidance Test"
echo "=========================================="
echo "This will test the new sigma-based style guidance"
echo "Parameters are conservative for safe testing"
echo "✓ Fixed tensor dimension issues"
echo "✓ Uses global + texture style matching"
echo ""
echo "Output: $OUT_DIR"
echo "Debug images: $DEBUG_OUTPUT_DIR"
echo "=========================================="

# Run the test
python inference.py \
    --init_mode $INIT_MODE \
    --gpu_ids $GPU_IDS \
    --prompts_file "$TEMP_PROMPTS" \
    --model_path "$MODEL_PATH" \
    --out_dir "$OUT_DIR" \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    --use_cfg_guidance \
    --cfg_guidance_scale $CFG_GUIDANCE_SCALE \
    --cfg_guidance_start_sigma $CFG_GUIDANCE_START_SIGMA \
    --cfg_guidance_end_sigma $CFG_GUIDANCE_END_SIGMA \
    --debug_output_dir "$DEBUG_OUTPUT_DIR"

# Save config
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/test_config.txt" << EOF
Quick Test Configuration
=======================
Style Guidance Scale: $CFG_GUIDANCE_SCALE
Sigma Range: $CFG_GUIDANCE_START_SIGMA → $CFG_GUIDANCE_END_SIGMA
Image Size: ${WIDTH}x${HEIGHT}
Seed: $SEED
EOF

# Cleanup
rm -f "$TEMP_PROMPTS"

echo ""
echo "=========================================="
echo "Test completed! Check results in: $OUT_DIR"
echo "Debug images show the generation process"
echo "=========================================="
