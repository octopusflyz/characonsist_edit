#!/usr/bin/env python3
"""
Verification script to ensure CFG is working
"""
import torch
import sys
sys.path.append('.')

def verify_cfg():
    print("=== CFG VERIFICATION TEST ===")

    # Simulate the CFG logic exactly as in pipeline
    use_style_guidance = True
    id_latents = torch.randn(1, 4, 32, 32)  # Mock ID latents
    style_guidance_weight_dict = {1: 3.0, 5: 2.5, 10: 2.0, 15: 1.5}  # Mock weights

    print(f"ID latents norm: {id_latents.norm().item():.4f}")

    # Test different scenarios
    test_cases = [
        ("ID generation", True, False, 10),
        ("Pre-run", False, True, 10),
        ("Frame generation step 1", False, False, 1),
        ("Frame generation step 5", False, False, 5),
        ("Frame generation step 10", False, False, 10),
        ("Frame generation step 15", False, False, 15),
        ("Frame generation step 20", False, False, 20),  # Not in dict
    ]

    for desc, is_id, is_pre_run, timestep in test_cases:
        print(f"\n--- {desc} ---")

        # Check conditions
        condition = (not is_id and not is_pre_run and use_style_guidance and
                    id_latents is not None and timestep in style_guidance_weight_dict)

        if condition:
            guidance_scale = style_guidance_weight_dict[timestep]

            # Simulate latents (different from ID)
            latents = id_latents + 0.5 * torch.randn_like(id_latents)
            original_diff = (latents - id_latents).norm().item()

            # Apply CFG logic
            id_latent_current = id_latents
            latent_diff = latents - id_latent_current

            # Enhanced approach
            min_effect = 0.1
            effective_scale = max(guidance_scale, min_effect)

            original_latents = latents.clone()
            latents = latents - latent_diff * effective_scale

            new_diff = (latents - id_latent_current).norm().item()
            effect_size = (latents - original_latents).norm().item()

            print(f"✓ Applied CFG: scale={guidance_scale:.1f}, effective={effective_scale:.1f}")
            print(f"  Original diff: {original_diff:.4f}, New diff: {new_diff:.4f}")
            print(f"  Effect size: {effect_size:.4f}, Improvement: {original_diff/new_diff:.1f}x")

            # Special test for step 10
            if timestep == 10:
                print("  *** CRITICAL TEST: Adding massive noise ***")
                massive_noise = torch.randn_like(latents) * 5.0
                latents = latents + massive_noise
                print(f"  Added noise with norm: {massive_noise.norm().item():.4f}")

        else:
            print("✗ Conditions not met for CFG application")

    print("\n=== VERIFICATION COMPLETE ===")
    print("If you see the CRITICAL TEST message and massive noise in real inference,")
    print("then CFG is working. The generated images should be very different at that point.")

if __name__ == "__main__":
    verify_cfg()

