#!/usr/bin/env python3
"""
Test complete CFG flow
"""
import torch
import sys
sys.path.append('.')

def test_cfg_flow():
    print("=== Testing Complete CFG Flow ===")

    # Simulate the complete inference flow
    from models.pipeline_characonsist import CharaConsistPipeline, get_style_guidance_weight

    # Create a mock pipeline with CFG enabled
    class MockCharaConsistPipeline:
        def __init__(self):
            self._use_style_guidance = True
            self._style_guidance_scale = 10.0
            self._style_guidance_start_step = 1
            self._style_guidance_end_step = 5
            self._style_guidance_weight_dict = get_style_guidance_weight(
                self._style_guidance_scale,
                self._style_guidance_start_step,
                self._style_guidance_end_step
            )
            self._id_latents = None

        def simulate_generation(self, is_id, is_pre_run, num_steps=10):
            print(f"\n--- Simulating {'ID' if is_id else 'Frame'} generation ({'pre-run' if is_pre_run else 'actual'}) ---")

            for i in range(num_steps):
                # Simulate denoising loop
                latents = torch.randn(1, 4096, 64)  # Mock latents

                # Check CFG condition
                condition = (not is_id and not is_pre_run and
                           self._use_style_guidance and
                           self._id_latents is not None and
                           i in self._style_guidance_weight_dict)

                if condition:
                    guidance_scale = self._style_guidance_weight_dict[i]
                    effective_scale = max(guidance_scale, 0.1)

                    # Simulate CFG application
                    id_latent_current = self._id_latents
                    latent_diff = latents - id_latent_current
                    original_latents = latents.clone()
                    latents = latents - latent_diff * effective_scale

                    diff_norm = latent_diff.norm().item()
                    effect_norm = (latents - original_latents).norm().item()

                    print(f"[CFG_APPLIED] Step {i}: scale={guidance_scale:.3f}, effective={effective_scale:.3f}, diff_norm={diff_norm:.4f}, effect_norm={effect_norm:.4f}")
                elif i < 3:  # Only show first few steps if not applied
                    print(f"[CFG_CHECK] Step {i}: condition={condition}, has_id_latents={self._id_latents is not None}")

                # Simulate saving ID latents
                if is_id and i == num_steps - 1:
                    self._id_latents = latents.detach().clone()
                    print(f"[CFG_SAVED] ID latents saved, shape: {self._id_latents.shape}")

    # Test the flow
    pipeline = MockCharaConsistPipeline()

    # 1. ID generation
    pipeline.simulate_generation(is_id=True, is_pre_run=False, num_steps=5)

    # 2. Frame pre-run
    pipeline.simulate_generation(is_id=False, is_pre_run=True, num_steps=5)

    # 3. Frame actual generation
    pipeline.simulate_generation(is_id=False, is_pre_run=False, num_steps=5)

    print("\n=== CFG Flow Test Completed ===")

if __name__ == "__main__":
    test_cfg_flow()

