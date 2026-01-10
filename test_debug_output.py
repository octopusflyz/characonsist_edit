#!/usr/bin/env python3
"""
Test debug output functionality
"""
import torch
import sys
import os
sys.path.append('.')

def test_debug_output():
    print("=== Testing Debug Output Functionality ===")

    # Create a mock pipeline-like object
    class MockCharaConsistPipeline:
        def __init__(self):
            from models.pipeline_characonsist import get_style_guidance_weight
            self._use_style_guidance = True
            self._style_guidance_scale = 10.0
            self._style_guidance_start_step = 1
            self._style_guidance_end_step = 5
            self._style_guidance_weight_dict = get_style_guidance_weight(
                self._style_guidance_scale,
                self._style_guidance_start_step,
                self._style_guidance_end_step
            )
            self._id_latents = torch.randn(1, 4096, 64)
            self._debug_output_dir = "test_debug_output"

            # Create debug directory
            os.makedirs(self._debug_output_dir, exist_ok=True)

        def simulate_denoising_step(self, i, latents, is_id=False, is_pre_run=False):
            """Simulate one denoising step with debug output"""

            # Mock denoising logic
            noise = torch.randn_like(latents) * 0.1
            latents = latents + noise  # Simplified denoising

            # Debug: Save intermediate images every 10 steps
            if self._debug_output_dir and (i + 1) % 10 == 0:
                try:
                    # Mock image saving (in real implementation this would decode VAE)
                    debug_filename = f"{self._debug_output_dir}/step_{i+1:03d}_{'id' if is_id else 'frame'}_{'prerun' if is_pre_run else 'final'}.jpg"
                    # In real code, this would be actual image saving
                    print(f"[DEBUG] Would save intermediate image at step {i+1}: {debug_filename}")
                except Exception as e:
                    print(f"[DEBUG] Failed to save intermediate image at step {i+1}: {e}")

            # Apply CFG if conditions met
            if not is_id and not is_pre_run and self._use_style_guidance and self._id_latents is not None and i in self._style_guidance_weight_dict:
                guidance_scale = self._style_guidance_weight_dict[i]
                id_latent_current = self._id_latents
                latent_diff = latents - id_latent_current
                effective_scale = max(guidance_scale, 0.1)
                latents = latents - latent_diff * effective_scale
                print(f"[CFG_APPLIED] Step {i}: applied guidance with scale {guidance_scale:.3f}")

            return latents

    # Test the debug functionality
    pipeline = MockCharaConsistPipeline()

    print(f"Debug output directory: {pipeline._debug_output_dir}")
    print(f"CFG weight dict: {pipeline._style_guidance_weight_dict}")

    # Simulate denoising loop
    latents = torch.randn(1, 4096, 64)
    print("\n=== Simulating Denoising Loop ===")

    for i in range(50):  # 50 steps like in real inference
        latents = pipeline.simulate_denoising_step(i, latents, is_id=False, is_pre_run=False)

        # Only show some steps for brevity
        if (i + 1) % 10 == 0:
            print(f"Completed step {i+1}")

    print("\n=== Debug Output Test Completed ===")
    print(f"Check directory: {pipeline._debug_output_dir}")

if __name__ == "__main__":
    test_debug_output()
