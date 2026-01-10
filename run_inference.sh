#!/bin/bash

# CharaConsist Inference Script

# Model and hardware settings
INIT_MODE=2
GPU_IDS="3 4"
MODEL_PATH="/mnt/netdisk2/zhangyf/model/FLUX.1-dev"
OUT_DIR="results/test/two_people_v2_adain"

# Image settings
HEIGHT=1024
WIDTH=1024
SEED=2025

# Style consistency settings (legacy)
USE_STYLE_CONSISTENCY=false
STYLE_LAYERS="8"
STYLE_WEIGHT=100
STYLE_MAX_TIMESTEP=30

# Progressive Style Guidance settings
USE_CFG_GUIDANCE=true
CFG_GUIDANCE_SCALE=0.1   # Gentle influence for style guidance
CFG_GUIDANCE_START_SIGMA=0.8    # Start when noise is moderate
CFG_GUIDANCE_END_SIGMA=0.2      # End before details finalize
# Legacy parameters (kept for compatibility)
CFG_GUIDANCE_START_STEP=40
CFG_GUIDANCE_END_STEP=51

# Debug settings
DEBUG_OUTPUT_DIR="${OUT_DIR}/debug_steps"  # Save intermediate images every 10 steps

# Prompts from gen-fg_only.ipynb
#; a young black boy with straight brown hair, wearing a brown T-shirt and blue shorts
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
TEMP_PROMPTS="/tmp/prompts_$$.txt"
> "$TEMP_PROMPTS"

# Create debug output directory if needed
if [ -n "$DEBUG_OUTPUT_DIR" ]; then
    mkdir -p "$DEBUG_OUTPUT_DIR"
fi

# Build prompts file
for i in "${!BG_PROMPTS[@]}"; do
    echo "${BG_PROMPTS[$i]}#${FG_PROMPT}#${ACT_PROMPTS[$i]}" >> "$TEMP_PROMPTS"
done

# Run inference
python inference.py \
    --init_mode $INIT_MODE \
    --gpu_ids $GPU_IDS \
    --prompts_file "$TEMP_PROMPTS" \
    --model_path "$MODEL_PATH" \
    --out_dir "$OUT_DIR" \
    --use_interpolate \
    --save_mask \
    --save_point_match \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    ${USE_STYLE_CONSISTENCY:+--use_style_consistency} \
    --style_layers $STYLE_LAYERS \
    --style_weight $STYLE_WEIGHT \
    --style_max_timestep $STYLE_MAX_TIMESTEP \
    ${USE_CFG_GUIDANCE:+--use_cfg_guidance} \
    --cfg_guidance_scale $CFG_GUIDANCE_SCALE \
    --cfg_guidance_start_step $CFG_GUIDANCE_START_STEP \
    --cfg_guidance_end_step $CFG_GUIDANCE_END_STEP \
    --cfg_guidance_start_sigma $CFG_GUIDANCE_START_SIGMA \
    --cfg_guidance_end_sigma $CFG_GUIDANCE_END_SIGMA \
    --debug_output_dir "$DEBUG_OUTPUT_DIR"

# Save prompts info for visualization
mkdir -p "$OUT_DIR"
cat > "$OUT_DIR/prompts.txt" << EOF
Background Prompts:
$(printf '%s\n' "${BG_PROMPTS[@]}")

Foreground Prompt:
$FG_PROMPT

Action Prompts:
$(printf '%s\n' "${ACT_PROMPTS[@]}")

Configuration:
- Model: $MODEL_PATH
- Init Mode: $INIT_MODE
- GPUs: $GPU_IDS
- Resolution: ${WIDTH}x${HEIGHT}
- Seed: $SEED
- Use Interpolate: Yes
- Save Mask: Yes
- Save Point Match: Yes
- Style Consistency: ${USE_STYLE_CONSISTENCY:-No}
- Style Layers: ${STYLE_LAYERS:-N/A}
- Style Weight: ${STYLE_WEIGHT:-N/A}
- Style Max Timestep: ${STYLE_MAX_TIMESTEP:-N/A}
- CFG Guidance: ${USE_CFG_GUIDANCE:-No}
- CFG Scale: ${CFG_GUIDANCE_SCALE:-N/A}
- CFG Start Sigma: ${CFG_GUIDANCE_START_SIGMA:-N/A}
- CFG End Sigma: ${CFG_GUIDANCE_END_SIGMA:-N/A}
- CFG Start Step: ${CFG_GUIDANCE_START_STEP:-N/A}
- CFG End Step: ${CFG_GUIDANCE_END_STEP:-N/A}
EOF

# Cleanup
rm -f "$TEMP_PROMPTS"

echo "Results saved to: $OUT_DIR"
echo "Point match data: $OUT_DIR/point_match_data/"
echo "Prompts config: $OUT_DIR/prompts.txt"
echo "Debug intermediate images: $DEBUG_OUTPUT_DIR"
