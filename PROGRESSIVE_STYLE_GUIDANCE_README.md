# Progressive Style Guidance for CharaConsist

This directory contains the new progressive style guidance system that replaces the traditional "hard overlay" CFG approach with a more sophisticated, FLUX-compatible style transfer method.

## 🎯 Problem Solved

The original CFG implementation used simple latent interpolation (`latents = latents - (latents - id_latents) * scale`), which caused:
- **Hard overlay effects** where ID images were directly superimposed
- **Content and style mixing** - changing both character poses and artistic style simultaneously
- **FLUX incompatibility** - ignoring the diffusion model's sigma-based scheduling

## ✅ New Approach

The progressive style guidance system:
- **Separates content and style** using statistical matching
- **Respects FLUX's diffusion process** with sigma-based scheduling
- **Uses foreground masking** to only affect character regions
- **Applies guidance gradually** based on noise levels

### How It Works

1. **Early Stage (σ > 0.8)**: Match global color/brightness statistics
2. **Mid Stage (0.3 < σ < 0.8)**: Add local texture constraints
3. **Late Stage (σ < 0.3)**: Reduce constraints for detail generation

## 🚀 Quick Start Scripts

### 1. Test Script (Recommended for first run)
```bash
./test_progressive_style.sh
```
- Uses conservative parameters
- Small 512x512 images for quick testing
- Includes debug output to see the process

### 2. Full Production Script
```bash
./run_progressive_style_guidance.sh
```
- Production-ready parameters
- 1024x1024 high-resolution output
- Multiple prompt variations
- Complete configuration saving

### 3. Comparison Script
```bash
./compare_style_methods.sh
```
- Runs baseline, traditional CFG, and progressive guidance
- Same prompts and seeds for fair comparison
- Generates comparison analysis

## 📊 Key Parameters

### Sigma-Based Scheduling (New)
```bash
--cfg_guidance_start_sigma 0.9  # Start when noise is high
--cfg_guidance_end_sigma 0.2    # End before details finalize
```

### Guidance Strength
```bash
--cfg_guidance_scale 0.25       # Overall strength (0.1-0.5 recommended)
```

## 🔍 Understanding the Debug Output

The debug images show the generation process:
- **Early steps**: Noise reduction with global style matching
- **Mid steps**: Structure formation with texture constraints
- **Late steps**: Detail refinement with reduced guidance

Look for:
- Character consistency between ID and frame images
- Style transfer without "hard overlay" artifacts
- Natural scene composition preservation

## 🎨 Expected Results

With progressive style guidance, you should see:
- ✅ **Character appearance consistency** (same art style, colors)
- ✅ **Pose/action preservation** (different poses maintained)
- ✅ **Natural style blending** (no harsh transitions)
- ✅ **Background independence** (scenes adapt naturally)

## 🔧 Customization

### For Stronger Style Transfer
```bash
--cfg_guidance_scale 0.4
--cfg_guidance_start_sigma 0.95
```

### For Gentler Influence
```bash
--cfg_guidance_scale 0.15
--cfg_guidance_end_sigma 0.4
```

### For Different Timing
```bash
--cfg_guidance_start_sigma 0.7  # Start later
--cfg_guidance_end_sigma 0.1    # Continue longer
```

## 📁 File Structure

```
results/
├── test/
│   ├── quick_progressive_test/     # Quick test results
│   │   ├── debug/                   # Step-by-step images
│   │   ├── 0.jpg                   # Frame image
│   │   ├── id.jpg                  # ID image
│   │   └── test_config.txt         # Parameters used
│   └── progressive_style_guidance/ # Full test results
└── comparison_YYYYMMDD_HHMMSS/     # Comparison results
    ├── baseline/
    ├── traditional_cfg/
    ├── progressive_style/
    └── COMPARISON_README.md
```

## 🐛 Troubleshooting

### If style transfer is too weak:
- Increase `--cfg_guidance_scale`
- Adjust sigma range to start earlier/last longer

### If results look unnatural:
- Decrease `--cfg_guidance_scale`
- Check foreground masks are working properly

### If generation fails:
- Verify model path exists
- Check GPU memory availability
- Try with smaller resolution first

## 🔬 Technical Details

The system implements:
- **Adaptive Instance Normalization** for global style matching
- **Gram matrix constraints** for local texture preservation
- **Sigma-based scheduling** aligned with FLUX's diffusion process
- **Foreground masking** to isolate character regions
- **Gradient-based optimization** for texture constraints

This approach ensures style transfer happens at the appropriate diffusion stages while preserving the generative capabilities of FLUX.
