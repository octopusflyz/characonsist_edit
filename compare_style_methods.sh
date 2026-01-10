#!/bin/bash

# Comparison Script: Traditional CFG vs Progressive Style Guidance

BASE_DIR="results/comparison_$(date +%Y%m%d_%H%M%S)"
echo "Running comparison test in: $BASE_DIR"

# Common settings
MODEL_PATH="/mnt/netdisk2/zhangyf/model/FLUX.1-dev"
HEIGHT=512
WIDTH=512
SEED=42
INIT_MODE=0
GPU_IDS="0"

# Test prompt
BG_PROMPT="in a simple outdoor setting,"
FG_PROMPT="a cartoon character with blue hair, "
ACT_PROMPT="standing and waving, front view"

# Create temporary prompts file
TEMP_PROMPTS="/tmp/compare_prompts_$$.txt"
echo "${BG_PROMPT}#${FG_PROMPT}#${ACT_PROMPT}" > "$TEMP_PROMPTS"

echo "=========================================="
echo "Style Guidance Method Comparison"
echo "=========================================="
echo "Testing traditional CFG vs progressive style guidance"
echo "Same prompt, same seed, different guidance methods"
echo ""

# Test 1: No guidance (baseline)
echo "1. Testing baseline (no style guidance)..."
BASELINE_DIR="$BASE_DIR/baseline"
DEBUG_DIR="$BASELINE_DIR/debug"

python inference.py \
    --init_mode $INIT_MODE \
    --gpu_ids $GPU_IDS \
    --prompts_file "$TEMP_PROMPTS" \
    --model_path "$MODEL_PATH" \
    --out_dir "$BASELINE_DIR" \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    --debug_output_dir "$DEBUG_DIR"

echo "✓ Baseline completed"

# Test 2: Traditional CFG (hard overlay - what you didn't want)
echo "2. Testing traditional CFG guidance (step-based)..."
TRADITIONAL_DIR="$BASE_DIR/traditional_cfg"
DEBUG_DIR="$TRADITIONAL_DIR/debug"

python inference.py \
    --init_mode $INIT_MODE \
    --gpu_ids $GPU_IDS \
    --prompts_file "$TEMP_PROMPTS" \
    --model_path "$MODEL_PATH" \
    --out_dir "$TRADITIONAL_DIR" \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    --use_cfg_guidance \
    --cfg_guidance_scale 0.5 \
    --cfg_guidance_start_step 10 \
    --cfg_guidance_end_step 40 \
    --debug_output_dir "$DEBUG_DIR"

echo "✓ Traditional CFG completed"

# Test 3: New Progressive Style Guidance (what you wanted)
echo "3. Testing progressive style guidance (sigma-based)..."
PROGRESSIVE_DIR="$BASE_DIR/progressive_style"
DEBUG_DIR="$PROGRESSIVE_DIR/debug"

python inference.py \
    --init_mode $INIT_MODE \
    --gpu_ids $GPU_IDS \
    --prompts_file "$TEMP_PROMPTS" \
    --model_path "$MODEL_PATH" \
    --out_dir "$PROGRESSIVE_DIR" \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    --use_cfg_guidance \
    --cfg_guidance_scale 0.25 \
    --cfg_guidance_start_sigma 0.8 \
    --cfg_guidance_end_sigma 0.2 \
    --debug_output_dir "$DEBUG_DIR"

echo "✓ Progressive style guidance completed"

# Create comparison summary
SUMMARY_FILE="$BASE_DIR/COMPARISON_README.md"
cat > "$SUMMARY_FILE" << EOF
# Style Guidance Method Comparison

This comparison demonstrates the difference between traditional CFG and the new progressive style guidance system.

## Test Setup
- **Prompt**: ${BG_PROMPT} + ${FG_PROMPT} + ${ACT_PROMPT}
- **Resolution**: ${WIDTH}x${HEIGHT}
- **Seed**: $SEED
- **Model**: FLUX.1-dev

## Methods Tested

### 1. Baseline (No Style Guidance)
- **Directory**: baseline/
- **Description**: Standard generation without any style constraints
- **Expected**: Natural generation, may vary from ID image style

### 2. Traditional CFG (Step-based)
- **Directory**: traditional_cfg/
- **Parameters**:
  - Scale: 0.5
  - Start Step: 10
  - End Step: 40
- **Description**: Direct latent interpolation toward ID latents
- **Issue**: Can cause "hard overlay" effect, mixing content and style

### 3. Progressive Style Guidance (Sigma-based)
- **Directory**: progressive_style/
- **Parameters**:
  - Scale: 0.25
  - Start Sigma: 0.8
  - End Sigma: 0.2
- **Description**: Gradual style transfer using statistical matching
- **Benefits**:
  - Preserves content (object shapes, poses)
  - Transfers style (colors, textures) selectively
  - Respects FLUX diffusion process
  - Uses foreground masking

## Expected Results

The progressive style guidance should:
1. **Maintain character consistency** in appearance and pose
2. **Transfer artistic style** from ID to frame images
3. **Avoid hard overlay artifacts** visible in traditional CFG
4. **Preserve scene composition** while adapting character style

## Debug Images

Each test directory contains debug images showing the generation process:
- \`debug/step_XXX_frame.jpg\`: Frame generation steps
- \`debug/step_XXX_id.jpg\`: ID image (if applicable)

Compare the evolution to see how different guidance methods affect the generation process.

## Next Steps

1. Review the generated images visually
2. Adjust parameters based on results:
   - Increase \`cfg_guidance_scale\` for stronger style transfer
   - Adjust sigma range for different timing
   - Modify prompts for better testing
3. Run with different seeds to test consistency
EOF

# Cleanup
rm -f "$TEMP_PROMPTS"

echo ""
echo "=========================================="
echo "Comparison completed!"
echo "Results: $BASE_DIR"
echo "Summary: $BASE_DIR/COMPARISON_README.md"
echo ""
echo "Directories:"
echo "  - Baseline: $BASE_DIR/baseline/"
echo "  - Traditional CFG: $BASE_DIR/traditional_cfg/"
echo "  - Progressive Style: $BASE_DIR/progressive_style/"
echo "=========================================="
