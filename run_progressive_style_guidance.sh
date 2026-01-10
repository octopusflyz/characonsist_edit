#!/bin/bash

# Progressive Style Guidance Inference Script for CharaConsist
# Uses the new sigma-based progressive style guidance system

# Model and hardware settings
INIT_MODE=2
GPU_IDS="3 4"
MODEL_PATH="/mnt/netdisk2/zhangyf/model/FLUX.1-dev"
OUT_DIR="results/test/progressive_style_guidance"

# Image settings
HEIGHT=1024
WIDTH=1024
SEED=2025

# Progressive Style Guidance settings
USE_CFG_GUIDANCE=true
CFG_GUIDANCE_SCALE=0.25           # Overall guidance strength (gentle)
CFG_GUIDANCE_START_SIGMA=0.9      # Start when noise is high (early stage)
CFG_GUIDANCE_END_SIGMA=0.2        # End before details are finalized (late stage)

# Legacy parameters (kept for compatibility but not used in new system)
CFG_GUIDANCE_START_STEP=1
CFG_GUIDANCE_END_STEP=50

# Debug settings
DEBUG_OUTPUT_DIR="${OUT_DIR}/debug_steps"  # Save intermediate images every 10 steps

# Test prompts - you can modify these directly in the script
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
TEMP_PROMPTS="/tmp/progressive_style_prompts_$$.txt"
> "$TEMP_PROMPTS"

# Create debug output directory if needed
if [ -n "$DEBUG_OUTPUT_DIR" ]; then
    mkdir -p "$DEBUG_OUTPUT_DIR"
fi

# Build prompts file
for i in "${!BG_PROMPTS[@]}"; do
    echo "${BG_PROMPTS[$i]}#${FG_PROMPT}#${ACT_PROMPTS[$i]}" >> "$TEMP_PROMPTS"
done

echo "========================================"
echo "Progressive Style Guidance Inference"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUT_DIR"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Style Guidance: ${USE_CFG_GUIDANCE:-false}"
echo "Guidance Scale: $CFG_GUIDANCE_SCALE"
echo "Sigma Range: $CFG_GUIDANCE_START_SIGMA → $CFG_GUIDANCE_END_SIGMA"
echo "Debug Output: ${DEBUG_OUTPUT_DIR:-None}"
echo "========================================"

# Run inference with progressive style guidance
python inference.py \
    --init_mode $INIT_MODE \
    --gpu_ids $GPU_IDS \
    --prompts_file "$TEMP_PROMPTS" \
    --model_path "$MODEL_PATH" \
    --out_dir "$OUT_DIR" \
    --use_interpolate \
    --share_bg \
    --save_mask \
    --save_point_match \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    ${USE_CFG_GUIDANCE:+--use_cfg_guidance} \
    --cfg_guidance_scale $CFG_GUIDANCE_SCALE \
    --cfg_guidance_start_step $CFG_GUIDANCE_START_STEP \
    --cfg_guidance_end_step $CFG_GUIDANCE_END_STEP \
    --cfg_guidance_start_sigma $CFG_GUIDANCE_START_SIGMA \
    --cfg_guidance_end_sigma $CFG_GUIDANCE_END_SIGMA \
    --debug_output_dir "$DEBUG_OUTPUT_DIR"

# Save configuration for reference
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/config.txt" << EOF
Progressive Style Guidance Configuration
========================================

Background Prompts:
$(printf '  %s\n' "${BG_PROMPTS[@]}")

Foreground Prompt:
  $FG_PROMPT

Action Prompts:
$(printf '  %s\n' "${ACT_PROMPTS[@]}")

Model Settings:
  Model Path: $MODEL_PATH
  Init Mode: $INIT_MODE
  GPUs: $GPU_IDS
  Resolution: ${WIDTH}x${HEIGHT}
  Seed: $SEED

Style Guidance Settings:
  Progressive Style Guidance: ${USE_CFG_GUIDANCE:-false}
  Guidance Scale: $CFG_GUIDANCE_SCALE
  Start Sigma: $CFG_GUIDANCE_START_SIGMA
  End Sigma: $CFG_GUIDANCE_END_SIGMA
  Legacy Start Step: $CFG_GUIDANCE_START_STEP
  Legacy End Step: $CFG_GUIDANCE_END_STEP

Processing Options:
  Use Interpolate: Yes
  Share Background: Yes
  Save Mask: Yes
  Save Point Match: Yes
  Debug Output: ${DEBUG_OUTPUT_DIR:-None}

Notes:
- Progressive guidance uses sigma-based scheduling instead of step-based
- Early stage (high sigma): Global color/brightness matching
- Mid stage: Local texture constraints
- Late stage: Reduced constraints for detail generation
- Foreground masking ensures style transfer only affects characters
EOF

# Cleanup
rm -f "$TEMP_PROMPTS"

echo ""
echo "========================================"
echo "Inference completed!"
echo "Results saved to: $OUT_DIR"
echo "Configuration: $OUT_DIR/config.txt"
echo "Point match data: $OUT_DIR/point_match_data/"
echo "Debug intermediate images: $DEBUG_OUTPUT_DIR"
echo "========================================"
