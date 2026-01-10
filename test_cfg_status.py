#!/usr/bin/env python3
"""
Test CFG status in pipeline
"""
import torch
import sys
sys.path.append('.')

def test_cfg_status():
    print("Testing CFG status...")

    # Create a minimal pipeline-like object
    class MockPipeline:
        def __init__(self):
            from models.pipeline_characonsist import get_style_guidance_weight
            self._use_style_guidance = True
            self._id_latents = torch.randn(1, 4096, 64)  # Mock ID latents
            self._style_guidance_weight_dict = get_style_guidance_weight(10.0, 1, 5)
            print(f"Mock pipeline created with:")
            print(f"  _use_style_guidance: {self._use_style_guidance}")
            print(f"  _id_latents shape: {self._id_latents.shape}")
            print(f"  _style_guidance_weight_dict keys: {list(self._style_guidance_weight_dict.keys())}")

    # Test the CFG logic exactly as in pipeline
    pipeline = MockPipeline()

    # Simulate different generation stages
    test_cases = [
        ("ID generation", True, False, 5),
        ("Pre-run frame", False, True, 5),
        ("Actual frame gen step 1", False, False, 1),
        ("Actual frame gen step 5", False, False, 5),
        ("Actual frame gen step 10", False, False, 10),  # Not in dict
    ]

    for desc, is_id, is_pre_run, timestep in test_cases:
        print(f"\n--- {desc} ---")
        condition = (not is_id and not is_pre_run and
                    pipeline._use_style_guidance and
                    pipeline._id_latents is not None and
                    timestep in pipeline._style_guidance_weight_dict)

        print(f"Condition met: {condition}")

        if condition:
            guidance_scale = pipeline._style_guidance_weight_dict[timestep]
            print(f"Would apply CFG with scale: {guidance_scale}")
        else:
            print("CFG not applied")

    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_cfg_status()

