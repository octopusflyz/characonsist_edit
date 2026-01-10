#!/bin/bash

# CharaConsist Inference Script

# Model and hardware settings
INIT_MODE=2
GPU_IDS="3 4"
MODEL_PATH="/mnt/netdisk2/zhangyf/model/FLUX.1-dev"  # Update this path to your actual model location
OUT_DIR="results/multi_fg_test_final"

# Image settings
HEIGHT=1024
WIDTH=1024
SEED=2025

# Prompts from gen-fg_only.ipynb
# Multi-subject test
BG_PROMPTS=(
    "in a colorful theme park, roller coasters and amusement rides in the background,"
    "in an arcade, flashing lights and game machines in the background,"
    "in a fantasy-themed park, castles and fairy tale characters in the background,"
)
# Multi-subject foreground prompts (separated by #)
FG_PROMPTS=(
    "a American girl with long brown hair, wearing a blue dress and a white hat"
    "a Chinese girl with long black hair, wearing a red dress and a blue hat"
)
ACT_PROMPTS=(
    "riding a roller coaster together, excited expressions, front view"
    "playing arcade games together, focused expressions, side view"
    "posing with a costumed character, happy expressions, front view"
)

# Create temporary prompts file
TEMP_PROMPTS="/tmp/prompts_$$.txt"
> "$TEMP_PROMPTS"

# Build prompts file with multi-subject support
for i in "${!BG_PROMPTS[@]}"; do
    fg_combined=""
    for fg in "${FG_PROMPTS[@]}"; do
        fg_combined+="${fg}#"
    done
    fg_combined=${fg_combined%#}  # 移除最后一个#
    echo "${BG_PROMPTS[$i]}#${fg_combined}#${ACT_PROMPTS[$i]}" >> "$TEMP_PROMPTS"
done
echo "" >> "$TEMP_PROMPTS"  # Add empty line to mark end of group

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
