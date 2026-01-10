#!/usr/bin/env python3
"""
Test CFG state persistence
"""
import torch
import sys
sys.path.append('.')

def test_cfg_state():
    print("=== Testing CFG State Persistence ===")

    # Mock pipeline class to test state management
    class MockPipeline:
        def __init__(self):
            pass

        def setup_cfg(self, use_style_guidance, style_guidance_scale, style_guidance_start_step, style_guidance_end_step):
            from models.pipeline_characonsist import get_style_guidance_weight

            # Simulate the current logic
            cfg_key = (use_style_guidance, style_guidance_scale, style_guidance_start_step, style_guidance_end_step)
            if not hasattr(self, '_cfg_initialized') or self._cfg_initialized != cfg_key:
                self._use_style_guidance = use_style_guidance
                self._style_guidance_scale = style_guidance_scale
                self._style_guidance_weight_dict = get_style_guidance_weight(
                    style_guidance_scale, style_guidance_start_step, style_guidance_end_step)
                # Only reset _id_latents if CFG parameters changed
                if not hasattr(self, '_cfg_initialized'):
                    self._id_latents = None
                self._cfg_initialized = cfg_key
                print(f"[MOCK] CFG initialized: {cfg_key}")
            else:
                print(f"[MOCK] CFG already initialized, keeping state")

            print(f"[MOCK] Current state: use_guidance={self._use_style_guidance}, id_latents={getattr(self, '_id_latents', 'UNDEFINED')}")

        def save_id_latents(self):
            self._id_latents = torch.randn(1, 4096, 64)
            print(f"[MOCK] Saved ID latents: {self._id_latents.shape}")

        def check_cfg_conditions(self, is_id, is_pre_run, timestep):
            condition = (not is_id and not is_pre_run and
                        self._use_style_guidance and
                        self._id_latents is not None and
                        timestep in self._style_guidance_weight_dict)
            print(f"[MOCK] Step {timestep}: condition={condition}")
            return condition

    # Test the state management
    pipeline = MockPipeline()

    # First call - should initialize CFG
    print("\n--- First CFG setup ---")
    pipeline.setup_cfg(True, 1000.0, 40, 51)

    # Save ID latents
    pipeline.save_id_latents()

    # Second call with same parameters - should NOT reinitialize
    print("\n--- Second CFG setup (same params) ---")
    pipeline.setup_cfg(True, 1000.0, 40, 51)

    # Third call with different parameters - should reinitialize but keep latents
    print("\n--- Third CFG setup (different params) ---")
    pipeline.setup_cfg(True, 2000.0, 40, 51)

    # Test conditions
    print("\n--- Testing CFG conditions ---")
    pipeline.check_cfg_conditions(False, False, 40)  # Should be True
    pipeline.check_cfg_conditions(False, False, 10)  # Should be False (not in dict)
    pipeline.check_cfg_conditions(True, False, 40)   # Should be False (is_id=True)

    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_cfg_state()
