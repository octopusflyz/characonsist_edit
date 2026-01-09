#!/bin/bash

# CharaConsist Inference Script

# Model and hardware settings
INIT_MODE=2
GPU_IDS="0 1"
MODEL_PATH="/mnt/netdisk2/zhangyf/model/FLUX.1-dev"
OUT_DIR="results/one_people_v1_ada_tome"

# Image settings
HEIGHT=1024
WIDTH=1024
SEED=2025

# Prompts from gen-fg_only.ipynb
#; a young black boy with straight brown hair, wearing a brown T-shirt and blue shorts
BG_PROMPTS=(
    "in a colorful theme park, roller coasters and amusement rides in the background,"
    "in an arcade, flashing lights and game machines in the background,"
    "in a fantasy-themed park, castles and fairy tale characters in the background,"
)
FG_PROMPT="a American girl with long brown hair, wearing a blue dress, "
ACT_PROMPTS=(
    "riding a roller coaster, excited expression, front view"
    "playing an arcade game, focused expression, side view"
    "posing with a costumed character, happy expression, front view"
)

# Create temporary prompts file
TEMP_PROMPTS="/tmp/prompts_$$.txt"
> "$TEMP_PROMPTS"

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
    --seed $SEED

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
EOF

# Cleanup
rm -f "$TEMP_PROMPTS"

echo "Results saved to: $OUT_DIR"
echo "Point match data: $OUT_DIR/point_match_data/"
echo "Prompts config: $OUT_DIR/prompts.txt"
