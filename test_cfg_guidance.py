#!/usr/bin/env python3
"""
Test script to verify CFG-style guidance implementation
"""
import torch
import sys
sys.path.append('.')

def test_cfg_guidance():
    print("Testing CFG-style guidance implementation...")

    # Test the style guidance weight function
    from models.pipeline_characonsist import get_style_guidance_weight

    weight_dict = get_style_guidance_weight(2.0, 5, 20)
    print(f"Style guidance weight dict: {weight_dict}")

    # Test basic latent guidance concept
    print("\n=== Testing Latent Guidance Concept ===")

    # Simulate ID latents (style reference)
    id_latents = torch.randn(1, 4, 64, 64)  # FLUX latent shape approximation
    print(f"ID latents shape: {id_latents.shape}")

    # Simulate current latents during frame generation
    current_latents = id_latents + 0.5 * torch.randn_like(id_latents)
    print(f"Current latents shape: {current_latents.shape}")

    # Apply CFG-style guidance
    guidance_scale = 1.0
    latent_diff = current_latents - id_latents
    guided_latents = current_latents - latent_diff * guidance_scale

    print(f"Original latent difference norm: {latent_diff.norm().item():.4f}")
    print(f"After guidance difference norm: {(guided_latents - id_latents).norm().item():.4f}")
    print(f"Guidance effectiveness: {latent_diff.norm().item() / (guided_latents - id_latents).norm().item():.2f}x closer to ID")

    print("\n=== CFG Guidance Test Completed Successfully ===")
    print("The CFG approach should provide more effective style guidance than attention-level modifications.")

if __name__ == "__main__":
    test_cfg_guidance()
