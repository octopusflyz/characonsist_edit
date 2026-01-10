#!/usr/bin/env python3
"""
Test script to verify CFG pipeline integration
"""
import torch
import sys
sys.path.append('.')

def test_cfg_pipeline():
    print("Testing CFG pipeline integration...")

    # Create a minimal mock pipeline
    class MockPipeline:
        def __init__(self):
            self._use_style_guidance = True
            self._id_latents = torch.randn(1, 4, 32, 32)
            from models.pipeline_characonsist import get_style_guidance_weight
            self._style_guidance_weight_dict = get_style_guidance_weight(1.0, 1, 10)

        def test_cfg_logic(self, is_id, is_pre_run, timestep):
            # Simulate the CFG logic from pipeline
            if not is_id and not is_pre_run and self._use_style_guidance and self._id_latents is not None and timestep in self._style_guidance_weight_dict:
                guidance_scale = self._style_guidance_weight_dict[timestep]

                # Mock latents
                latents = torch.randn_like(self._id_latents) + 0.1 * torch.randn_like(self._id_latents)

                # Apply guidance
                id_latent_current = self._id_latents
                latent_diff = latents - id_latent_current
                original_latents = latents.clone()
                latents = latents - latent_diff * guidance_scale

                diff_before = latent_diff.norm().item()
                diff_after = (latents - id_latent_current).norm().item()

                print(f"[CFG TEST] Timestep {timestep}: applied guidance with scale {guidance_scale:.3f}")
                print(f"[CFG TEST] Diff before: {diff_before:.4f}, after: {diff_after:.4f}")
                print(f"[CFG TEST] Improvement: {diff_before/max(diff_after, 1e-8):.1f}x")

                return True
            else:
                print(f"[CFG TEST] Timestep {timestep}: conditions not met")
                print(f"  is_id: {is_id}, is_pre_run: {is_pre_run}")
                print(f"  use_style_guidance: {self._use_style_guidance}")
                print(f"  has_id_latents: {self._id_latents is not None}")
                print(f"  timestep in dict: {timestep in self._style_guidance_weight_dict}")
                return False

    pipeline = MockPipeline()

    # Test ID generation (should not apply CFG)
    print("\n=== Testing ID Generation ===")
    pipeline.test_cfg_logic(is_id=True, is_pre_run=False, timestep=5)

    # Test pre-run (should not apply CFG)
    print("\n=== Testing Pre-run ===")
    pipeline.test_cfg_logic(is_id=False, is_pre_run=True, timestep=5)

    # Test frame generation (should apply CFG)
    print("\n=== Testing Frame Generation ===")
    for t in [1, 5, 10]:
        print(f"\n--- Timestep {t} ---")
        pipeline.test_cfg_logic(is_id=False, is_pre_run=False, timestep=t)

    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_cfg_pipeline()

