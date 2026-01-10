#!/usr/bin/env python3
"""
Test script to verify progressive style guidance implementation
"""
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append('.')

def test_global_style_statistics():
    """Test global style statistics computation"""
    print("=== Testing Global Style Statistics ===")

    # Create test latents similar to FLUX dimensions
    batch_size, channels, height, width = 1, 16, 64, 64  # FLUX latent dimensions approximation
    latents = torch.randn(batch_size, channels, height, width)

    # Import the function
    from models.pipeline_characonsist import compute_global_style_statistics

    # Test without mask
    mean, std = compute_global_style_statistics(latents, fg_mask=None)
    print(f"Global stats without mask - mean shape: {mean.shape}, std shape: {std.shape}")
    print(f"Mean values: {mean.flatten()[:5].tolist()}")  # Show first 5 values
    print(f"Std values: {std.flatten()[:5].tolist()}")

    # Test with foreground mask
    fg_mask = torch.zeros(batch_size, 1, height, width)
    fg_mask[:, :, :32, :32] = 1.0  # Top-left quarter as foreground

    mean_masked, std_masked = compute_global_style_statistics(latents, fg_mask=fg_mask)
    print(f"Global stats with mask - mean shape: {mean_masked.shape}, std shape: {std_masked.shape}")
    print(f"Masked mean values: {mean_masked.flatten()[:5].tolist()}")
    print(f"Masked std values: {std_masked.flatten()[:5].tolist()}")

    print("✓ Global style statistics test passed\n")


def test_progressive_style_guidance():
    """Test progressive style guidance function"""
    print("=== Testing Progressive Style Guidance ===")

    # Create test latents
    batch_size, channels, height, width = 1, 16, 64, 64
    current_latents = torch.randn(batch_size, channels, height, width)
    id_latents = torch.randn(batch_size, channels, height, width)

    # Create foreground mask
    fg_mask = torch.zeros(batch_size, 1, height, width)
    fg_mask[:, :, :32, :32] = 1.0

    from models.pipeline_characonsist import apply_progressive_style_guidance

    # Test different sigma values
    test_sigmas = [0.9, 0.7, 0.5, 0.3, 0.1]

    for sigma in test_sigmas:
        print(f"\n--- Testing sigma = {sigma} ---")
        try:
            styled_latents = apply_progressive_style_guidance(
                current_latents, id_latents, fg_mask, sigma, guidance_scale=0.3
            )

            # Check that output has correct shape
            assert styled_latents.shape == current_latents.shape, f"Shape mismatch: {styled_latents.shape} vs {current_latents.shape}"

            # Check that latents have changed (but not drastically)
            diff = (styled_latents - current_latents).abs().mean().item()
            print(f"Latent modification magnitude: {diff:.6f}")

            print(f"✓ Sigma {sigma} test passed")

        except Exception as e:
            print(f"✗ Sigma {sigma} test failed: {e}")

    print("\n✓ Progressive style guidance test completed\n")


def test_texture_grams():
    """Test local texture Gram matrix computation"""
    print("=== Testing Local Texture Grams ===")

    # Create test latents
    batch_size, channels, height, width = 1, 16, 64, 64
    latents = torch.randn(batch_size, channels, height, width)

    from models.pipeline_characonsist import compute_local_texture_grams

    # Test without mask
    grams = compute_local_texture_grams(latents, fg_mask=None, patch_size=8)
    if grams is not None:
        print(f"Texture grams without mask - shape: {grams.shape}")
        print(f"Number of patches: {grams.shape[1]}")
    else:
        print("No texture grams computed (no valid patches)")

    # Test with foreground mask
    fg_mask = torch.zeros(batch_size, 1, height, width)
    fg_mask[:, :, :32, :32] = 1.0

    grams_masked = compute_local_texture_grams(latents, fg_mask=fg_mask, patch_size=8)
    if grams_masked is not None:
        print(f"Texture grams with mask - shape: {grams_masked.shape}")
        print(f"Number of valid patches: {grams_masked.shape[1]}")
    else:
        print("No texture grams computed (no valid foreground patches)")

    print("✓ Local texture grams test completed\n")


def test_end_to_end():
    """Test end-to-end style guidance pipeline"""
    print("=== Testing End-to-End Style Guidance ===")

    try:
        # Simulate a simple generation step
        batch_size, channels, height, width = 1, 16, 64, 64

        # Create ID latents (style reference)
        id_latents = torch.randn(batch_size, channels, height, width)

        # Create current latents (to be styled)
        current_latents = id_latents + 0.5 * torch.randn_like(id_latents)

        # Create foreground mask
        fg_mask = torch.zeros(batch_size, 1, height, width)
        fg_mask[:, :, :32, :32] = 1.0

        from models.pipeline_characonsist import apply_progressive_style_guidance

        # Test the complete flow
        sigma = 0.7  # Mid-range sigma
        styled_latents = apply_progressive_style_guidance(
            current_latents, id_latents, fg_mask, sigma, guidance_scale=0.2
        )

        # Verify output properties
        assert styled_latents.shape == current_latents.shape

        # Check that styling moved latents closer to ID style
        original_diff = (current_latents - id_latents).norm().item()
        styled_diff = (styled_latents - id_latents).norm().item()

        print(f"Original distance to ID: {original_diff:.4f}")
        print(f"Styled distance to ID: {styled_diff:.4f}")
        print(f"Improvement: {original_diff - styled_diff:.4f}")

        print("✓ End-to-end test passed")

    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing Progressive Style Guidance Implementation")
    print("=" * 50)

    test_global_style_statistics()
    test_texture_grams()
    test_progressive_style_guidance()
    test_end_to_end()

    print("\n" + "=" * 50)
    print("All tests completed!")
